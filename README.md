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

## What happened (seed 0, 1200 train / 600 test, held out)

| model | accuracy | avg attention pairs |
|-------|----------|---------------------|
| silo-only (always cheap) | 0.678 | 16 |
| plain-only (always expensive) | 0.780 | 81 |
| **`<ono>` router** | **0.877** | **49.1** |

Routing policy: **bag → silo 98.6%**, **order → silo 1.0%** (i.e. order → plain ~99%).

**Read it straight.** The router **beats both single-expert baselines** *and*
spends **~40% less** attention on average than always-plain. The accuracy edge is
**expert specialisation** — because the router sends bag examples to the silo head
and order examples to the plain head, each head trains almost entirely on its own
regime and gets clean at it, so together they beat a single "do-everything" plain
model trained at the same budget. The compute saving is **conditional
computation**: it only pays the plain `N²` cost when order actually matters.

## The one honest catch

A router can only route because **the regime is detectable in the input** — here,
a cue token carries it. Order-sensitivity is a property of the *task*, not always
of a single example in isolation; **if no cue carried it, no router could tell**,
and this one would be routing on nothing. The compute penalty is standard
cost-aware routing, not magic. Synthetic probes, tiny models, one seed, matched
budget — a clean demonstration, **not** a leaderboard.

## Verify first

```bash
python selftest.py    # gradient-checks the router-MoE backprop (with & without the compute penalty), + router invariants
python train.py       # retrain the router + both baselines on the mixed set -> results.json
```

The **gradient check is v5's honesty anchor**: the router-MoE — shared encoder,
two experts, a learned gate, a probability-level mixture, and the compute penalty
— really trains by gradient descent (analytic grads match numerical `< 1e-5`).

## Files

| File | Role |
|------|------|
| `model.py` | shared encoder + silo & plain heads + the `<ono>` router; hand-written, gradient-checked backward |
| `tasks.py` | the world, the content silo, the mixed-regime data + the regime cue |
| `train.py` | Adam training of the router + baselines → `results.json` |
| `selftest.py` | gradient checks (± compute penalty) + router/routing invariants |
| `results.json` | the trained results the page reports |
| `index.html` | the routing diagram + three-model results page + the verdict |

The pentalogy: [v1](https://davidwise01.github.io/transformer-silo/) build ·
[v2](https://davidwise01.github.io/transformer-silo-v2/) measure ·
[v3](https://davidwise01.github.io/transformer-silo-v3/) train one silo ·
[v4](https://davidwise01.github.io/transformer-silo-v4/) two in parallel · v5 route between them.

---
David Lee Wise / ROOT0 / TriPod LLC · CC-BY-ND-4.0
