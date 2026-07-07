#!/usr/bin/env python3
"""Honesty anchor for v5: gradient-check the hand-written router-MoE backprop
(with AND without the compute penalty) analytically vs numerical, and check the
router's forward invariants. If the gradients are wrong the training is theatre;
these checks are what make the results trustworthy. Run: python selftest.py"""
from __future__ import annotations
import numpy as np
from model import (init_params, loss_and_grad, encode, gate, softmax,
                   route_and_predict)
from tasks import make_world, mixed_dataset, moe_inputs, D, G, N, K

PASS = "PASS"; FAIL = "FAIL"
fails = 0
def check(name, cond, extra=""):
    global fails
    print(f"[{PASS if cond else FAIL}] {name}" + (f"  {extra}" if extra else ""))
    if not cond: fails += 1


def numgrad(f, P, key, eps=1e-6):
    g = np.zeros_like(P[key]); flat = P[key].reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps; lp = f()
        flat[i] = old - eps; lm = f()
        flat[i] = old
        g.reshape(-1)[i] = (lp - lm) / (2 * eps)
    return g


def grad_check(lam):
    world = make_world(seed=1)
    P = init_params(D, 8, G, seed=3)
    sX, sw, pX, pw = moe_inputs(world, 1, np.arange(N) % 4 + 1)
    y = 2
    _, g = loss_and_grad(P, sX, sw, pX, pw, y, lam=lam, cost_s=0.2, cost_p=1.0)
    worst = 0.0; worst_k = None
    for k in P:
        ng = numgrad(lambda: loss_and_grad(P, sX, sw, pX, pw, y, lam=lam,
                                           cost_s=0.2, cost_p=1.0)[0], P, k)
        rel = np.abs(g[k] - ng).max() / (np.abs(ng).max() + 1e-12)
        if rel > worst: worst, worst_k = rel, k
    return worst, worst_k


print("== gradient check: the router MoE (honesty anchor) ==")
for lam in (0.0, 0.15):
    worst, wk = grad_check(lam)
    check(f"analytic == numerical backprop (lam={lam})", worst < 1e-5,
          f"max rel err {worst:.2e} at {wk}")

print("\n== router forward invariants ==")
world = make_world(seed=0)
P = init_params(D, 8, G, seed=0)
sX, sw, pX, pw = moe_inputs(world, 0, np.arange(N) % 4)
q, rf = gate(P, pX)
check("gate is a distribution over 2 experts", abs(q.sum() - 1.0) < 1e-9 and (q >= 0).all(),
      f"q={q.round(3)}")
pred, which, pairs = route_and_predict(P, sX, sw, pX, pw)
check("hard route picks a valid expert", which in ("silo", "plain"))
check("silo route costs K^2, plain route costs (N+1)^2",
      (which == "silo" and pairs == K * K) or (which == "plain" and pairs == (N + 1) ** 2),
      f"which={which} pairs={pairs}")

# mixture at q -> matches manual mix; predictions are argmax of a real distribution
ps, _ = encode(P, sX, sw); pp, _ = encode(P, pX, pw)
Ls = ps @ P["Wcs"] + P["bcs"]; Lp = pp @ P["Wcp"] + P["bcp"]
Pmix = q[0] * softmax(Ls) + q[1] * softmax(Lp)
check("training mixture is a valid distribution", abs(Pmix.sum() - 1.0) < 1e-9)

# the compute penalty actually biases the gate toward the cheap silo
def mean_qsilo(lam, epochs=8):
    from train import train_moe
    ds = mixed_dataset(world, n_train=200, n_test=1, seed=7)
    Pm = train_moe(world, ds["train"], seed=0, epochs=epochs, lam=lam)
    qs = [gate(Pm, moe_inputs(world, r, t)[2])[0][0] for (r, t, _) in ds["train"]]
    return float(np.mean(qs))
q_no = mean_qsilo(0.0); q_pen = mean_qsilo(0.5)
check("compute penalty raises average q_silo (cost-aware routing)", q_pen > q_no,
      f"mean q_silo: lam=0 -> {q_no:.3f}, lam=0.5 -> {q_pen:.3f}")

print("\n" + ("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED"))
raise SystemExit(1 if fails else 0)
