# transformer-silo v5 — the `<ono>` router

v3 trained one silo against a plain transformer and found the honest edge: the
silo is cheap but **collapses on the order task** because it throws position away.
v4 ran two silos in parallel to buy some of it back. v5 asks the obvious question —
*"why not have a router? you can tell if the data is order-sensitive or not"* —
and builds it: a **router**, the `<ono>` **(order / no-order) gate**, that reads
the input and sends order-**in**sensitive examples to the cheap content silo and
order-**sensitive** ones to the plain, order-aware expert.

## The router (a tiny mixture-of-experts)

A shared transformer encoder feeds **two experts** — a **silo** head (over K
content intents; cheap, orderless) and a **plain** head (over N tokens +
positions; order-aware, expensive). A small **router** reads a summary of the
input and outputs a gate `q = [q_silo, q_plain]`. Training mixes the experts at
the probability level so the whole thing is differentiable and the router
co-trains with the experts:

```
P = q_silo · softmax(logits_silo) + q_plain · softmax(logits_plain)
```

At inference you **hard-route** to the preferred expert and pay only *that*
expert's attention — `K²` when the silo is chosen, `(N+1)²` when plain is. The
objective adds a small **compute penalty** `λ · (q_silo·cost_silo + q_plain·cost_plain)`
— standard cost-aware routing — that nudges the router to the cheap path when the
silo already suffices.

## The mixed-regime task

Each example is one of two regimes, and a **regime cue token** rides in the input
so the router *can* tell them apart:

- **bag** (order-**in**sensitive) — label = the most frequent group. The silo can do it.
- **order** (order-**sensitive**) — label = the first token's group. Only the order-aware expert can.

## What happened (seed 0, 1200 train / 600 test, held out; both arms to convergence)

| model | accuracy | avg attention pairs |
|-------|----------|---------------------|
| silo-only (always cheap) | 0.665 | 16 |
| plain-only (always expensive) | 0.917 | 81 |
| **`<ono>` router** | **0.937** | **49.0** |

Routing policy (seed 0): **bag → silo 100%**, **order → plain 100%** — the router
learns the regime cleanly. Per regime: silo-only and plain-only **tie on the bag
task** (0.915 each), the silo **collapses on order** (0.42, an unordered set can't
read position) while plain handles it (0.92) — which is why the cheap silo can't
stand alone.

**Read it straight — and check more than one seed.** The **robust** result is
**conditional computation**: the router spends **~40% less** attention on average
(49 vs 81 pairs) than always-plain, and it does this on **every** seed. Its
*accuracy* lands within a few points of the strong plain baseline — but the sign
of the gap is **not** guaranteed.

## Robustness (5 seeds) — the honest headline

Running `python sweep.py` (seeds 0–4, all to convergence):

| seed | silo | plain | router | router − plain | compute saved |
|------|------|-------|--------|----------------|---------------|
| 0 | 0.665 | 0.917 | 0.937 | **+2.0** | 40% |
| 1 | 0.612 | 0.948 | 0.995 | **+4.7** | 40% |
| 2 | 0.635 | 0.780 | 0.872 | **+9.2** | 40% |
| 3 | 0.592 | 0.940 | 0.978 | **+3.8** | 40% |
| 4 | 0.628 | 0.993 | 0.963 | **−3.0** | 39% |

So, stated plainly: the router **saves ~40% average compute on every seed** and
**always beats the always-cheap silo**, but versus the strong plain baseline it
**usually edges ahead (4 / 5 seeds, mean +3.3 pts) — and trails on 1 of 5.** The
small accuracy edge, when it appears, is **expert specialisation** (each head
trains almost only on its own regime); it is **not** a guaranteed accuracy win,
and this README does not claim one. The win worth banking is the compute.

## The honest caveats

- **Not a guaranteed accuracy win** — router − plain runs **−3 to +9 pts** across
  seeds (it trails plain on 1 of 5). The reliable, reproducible gain is the ~40%
  average-compute saving, plus reliably beating the cheap silo.
- **A router can only route because the regime is detectable in the input** — here,
  a cue token carries it. Order-sensitivity is a property of the *task*; **if no
  cue carried it, no router could tell**, and this one would be routing on nothing.
- **Slightly more capacity** — the router adds a second head + gate (**428 vs 386**
  params) over a *shared* encoder. Tiny, but noted, not hidden.
- The compute penalty is standard **cost-aware routing**, not magic. Synthetic
  probes, tiny models — a clean demonstration, **not** a leaderboard.

## Verify first

```bash
python selftest.py    # gradient-checks the router-MoE backprop (with & without the compute penalty), + router invariants
python train.py       # retrain the router + both baselines on the mixed set -> results.json (seed 0)
python sweep.py       # the 5-seed robustness sweep -> sweep.json
```

The **gradient check is v5's honesty anchor**: the router-MoE — shared encoder,
two experts, a learned gate, a probability-level mixture, and the compute penalty
— really trains by gradient descent (analytic grads match numerical `< 1e-5`).

## Files

| File | Role |
|------|------|
| `model.py` | shared encoder + silo & plain heads + the `<ono>` router; hand-written, gradient-checked backward |
| `tasks.py` | the world, the content silo, the mixed-regime data + the regime cue |
| `train.py` | Adam training of the router + baselines (to convergence) → `results.json` |
| `sweep.py` | the 5-seed robustness sweep → `sweep.json` |
| `selftest.py` | gradient checks (± compute penalty) + router/routing invariants |
| `results.json` | the seed-0 trained results the page reports |
| `index.html` | the routing diagram + three-model results page + the verdict |

The pentalogy: [v1](https://davidwise01.github.io/transformer-silo/) build ·
[v2](https://davidwise01.github.io/transformer-silo-v2/) measure ·
[v3](https://davidwise01.github.io/transformer-silo-v3/) train one silo ·
[v4](https://davidwise01.github.io/transformer-silo-v4/) two in parallel · v5 route between them.

---
David Lee Wise / ROOT0 / TriPod LLC · CC-BY-ND-4.0
