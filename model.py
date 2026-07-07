#!/usr/bin/env python3
"""v5 model: a ROUTER (the <ono> gate) over two experts.

A shared transformer encoder feeds two heads -- a SILO expert (over K content
intents; cheap, orderless) and a PLAIN expert (over N tokens + positions; order-
aware, expensive). A small router reads a summary of the input and outputs a gate
q = [q_silo, q_plain]. Training mixes the experts at the probability level:

    P = q_silo * softmax(logits_silo) + q_plain * softmax(logits_plain)

so it is fully differentiable and the router co-trains with the experts. At
inference you can HARD-route (take the argmax expert) and pay only that expert's
attention -- K^2 when the silo is chosen, N^2 when plain is. The router can only
learn to route if the input carries a signal of the regime; that caveat is the
honest heart of v5.

All backprop is hand-written and gradient-checked in selftest.py.
"""
from __future__ import annotations
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def init_params(d, h, C, seed=0):
    rng = np.random.default_rng(seed)
    s = 0.2
    def w(*shape): return rng.standard_normal(shape) * s
    return {
        "Wq": w(d, d), "Wk": w(d, d), "Wv": w(d, d), "Wo": w(d, d),
        "W1": w(d, h), "b1": np.zeros(h), "W2": w(h, d), "b2": np.zeros(d),
        "Wcs": w(d, C), "bcs": np.zeros(C),          # silo head
        "Wcp": w(d, C), "bcp": np.zeros(C),          # plain head
        "Wr": w(d, 2), "br": np.zeros(2),            # the <ono> router: [silo, plain]
    }


def _zeros_like(P):
    return {k: np.zeros_like(v) for k, v in P.items()}


# ---------- shared encoder: (X, w) -> pooled vector ----------
def encode(P, X, w=None):
    n, d = X.shape
    if w is None:
        w = np.ones(n)
    wn = w / w.sum()
    Q, K, V = X @ P["Wq"], X @ P["Wk"], X @ P["Wv"]
    S = (Q @ K.T) / np.sqrt(d)
    A = softmax(S, axis=1)
    Ctx = A @ V
    Attn = Ctx @ P["Wo"]
    Z1 = X + Attn
    Hpre = Z1 @ P["W1"] + P["b1"]
    H = np.maximum(0.0, Hpre)
    M = H @ P["W2"] + P["b2"]
    Z2 = Z1 + M
    p = (Z2 * wn[:, None]).sum(axis=0)
    return p, dict(X=X, K=K, V=V, Q=Q, A=A, Ctx=Ctx, Z1=Z1, Hpre=Hpre, H=H, wn=wn, d=d)


def encode_backward(P, cache, dp):
    g = {}
    dZ2 = np.outer(cache["wn"], dp)
    dZ1 = dZ2.copy()
    dM = dZ2
    g["W2"] = cache["H"].T @ dM
    g["b2"] = dM.sum(axis=0)
    dH = dM @ P["W2"].T
    dHpre = dH * (cache["Hpre"] > 0)
    g["W1"] = cache["Z1"].T @ dHpre
    g["b1"] = dHpre.sum(axis=0)
    dZ1 += dHpre @ P["W1"].T
    dAttn = dZ1
    g["Wo"] = cache["Ctx"].T @ dAttn
    dCtx = dAttn @ P["Wo"].T
    dA = dCtx @ cache["V"].T
    dV = cache["A"].T @ dCtx
    dS = cache["A"] * (dA - (dA * cache["A"]).sum(axis=1, keepdims=True))
    dS /= np.sqrt(cache["d"])
    dQ = dS @ cache["K"]
    dK = dS.T @ cache["Q"]
    g["Wq"] = cache["X"].T @ dQ
    g["Wk"] = cache["X"].T @ dK
    g["Wv"] = cache["X"].T @ dV
    return g


def _head(P, pooled, which):
    Wc = P["Wcs" if which == "s" else "Wcp"]
    bc = P["bcs" if which == "s" else "bcp"]
    return pooled @ Wc + bc


def gate(P, plain_X):
    rf = plain_X.mean(axis=0)                        # summary the router reads
    q = softmax(rf @ P["Wr"] + P["br"])
    return q, rf


# ---------- the router MoE: forward + loss + full gradient ----------
def loss_and_grad(P, silo_X, silo_w, plain_X, plain_w, y, lam=0.0, cost_s=0.25, cost_p=1.0):
    """Objective = cross-entropy of the mixture + lam * expected compute, where
    expected compute = q_silo*cost_s + q_plain*cost_p (costs are attention pairs,
    normalised). The compute term is cost-aware routing: it nudges the router to
    the CHEAP silo when the silo can already solve the example, and only spend on
    the plain expert when the cheap one is not good enough. Returns the full
    objective (CE + compute penalty) and its exact gradient."""
    ps, cs = encode(P, silo_X, silo_w)
    pp, cp = encode(P, plain_X, plain_w)
    Ls, Lp = _head(P, ps, "s"), _head(P, pp, "p")
    Psm, Ppm = softmax(Ls), softmax(Lp)
    q, rf = gate(P, plain_X)                          # [q_silo, q_plain]
    Pmix = q[0] * Psm + q[1] * Ppm
    # objective = cross-entropy + lam * expected compute (this IS what we optimise,
    # so the returned value matches the gradient -- self-consistent for grad-check)
    compute = float(q[0] * cost_s + q[1] * cost_p)
    loss = -np.log(Pmix[y] + 1e-12) + lam * compute

    g = _zeros_like(P)
    dPmix = np.zeros_like(Pmix)
    dPmix[y] = -1.0 / (Pmix[y] + 1e-12)

    # experts (probability mixture)
    dPsm = q[0] * dPmix
    dPpm = q[1] * dPmix
    dq = np.array([float((dPmix * Psm).sum()), float((dPmix * Ppm).sum())])
    dq += lam * np.array([cost_s, cost_p])           # cost-aware routing penalty

    # silo head + encoder
    dLs = Psm * (dPsm - (dPsm * Psm).sum())
    g["Wcs"] = np.outer(ps, dLs); g["bcs"] = dLs
    for k, v in encode_backward(P, cs, P["Wcs"] @ dLs).items():
        g[k] += v
    # plain head + encoder
    dLp = Ppm * (dPpm - (dPpm * Ppm).sum())
    g["Wcp"] = np.outer(pp, dLp); g["bcp"] = dLp
    for k, v in encode_backward(P, cp, P["Wcp"] @ dLp).items():
        g[k] += v
    # router (softmax over 2)
    dgl = q * (dq - (dq * q).sum())
    g["Wr"] = np.outer(rf, dgl); g["br"] = dgl
    return loss, g


def route_and_predict(P, silo_X, silo_w, plain_X, plain_w):
    """Hard route: pick the expert the gate prefers, return (pred, which, pairs)."""
    q, _ = gate(P, plain_X)
    if q[0] >= q[1]:
        ps, _ = encode(P, silo_X, silo_w)
        pred = int(np.argmax(_head(P, ps, "s")))
        return pred, "silo", silo_X.shape[0] ** 2
    pp, _ = encode(P, plain_X, plain_w)
    pred = int(np.argmax(_head(P, pp, "p")))
    return pred, "plain", plain_X.shape[0] ** 2
