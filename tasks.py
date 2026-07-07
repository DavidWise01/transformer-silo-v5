#!/usr/bin/env python3
"""The data world + the silo front-end for the v3 comparison.

A fixed embedding world: G latent groups, each with a prototype vector; V tokens,
each belonging to a group (embedding = its group's prototype + small noise), so
token embeddings genuinely cluster by group. Fixed positional embeddings let the
PLAIN arm see order; the SILO arm gets K k-means centroids of the bag (weighted
by cluster size) and has NO order.

Two standard synthetic probes (the kind used for architecture ablations):
  * PLURALITY  (order-INsensitive): label = the most frequent group in the bag.
                Frequency is preserved by size-weighted pooling, so both arms can
                learn it -- the silo at K^2 attention instead of N^2.
  * FIRST      (order-SENSITIVE):   label = the group of the FIRST token. Only an
                order-aware model can do better than chance; the silo, being an
                unordered set of intents, cannot -- and v3 reports that plainly.
"""
from __future__ import annotations
import numpy as np

D = 6          # embedding width
G = 4          # latent groups (= number of classes)
V = 16         # vocabulary size (4 tokens per group)
N = 8          # tokens per bag
K = 4          # silo intents


def make_world(seed=0):
    rng = np.random.default_rng(seed)
    protos = rng.standard_normal((G, D)) * 2.2            # well-separated group prototypes
    tok_group = np.array([g for g in range(G) for _ in range(V // G)])
    E = np.stack([protos[tok_group[t]] + rng.standard_normal(D) * 0.35 for t in range(V)])
    Ppos = rng.standard_normal((N, D)) * 0.5             # positional embeddings (plain arm)
    BinPos = rng.standard_normal((N, D)) * 0.9           # per-bin positional codes (position-silo)
    RegE = rng.standard_normal((2, D)) * 2.0             # regime cue: [0]=bag, [1]=order
    MarkPos = rng.standard_normal(D) * 0.5               # position code for the regime marker
    return {"protos": protos, "tok_group": tok_group, "E": E, "Ppos": Ppos,
            "BinPos": BinPos, "RegE": RegE, "MarkPos": MarkPos, "rng_seed": seed}


# ---------- the silo front-end: k-means on the bag's token embeddings ----------
def _seed_centroids(vs, k):
    chosen = [vs[0].copy()]
    while len(chosen) < k:
        d = np.min([np.sum((vs - c) ** 2, axis=1) for c in chosen], axis=0)
        chosen.append(vs[int(np.argmax(d))].copy())
    return np.stack(chosen)


def centrifuge(vs, k, max_spins=25):
    """Deterministic k-means (farthest-point seed). Returns (centroids, sizes)."""
    cen = _seed_centroids(vs, k)
    assign = np.argmin(((vs[:, None, :] - cen[None, :, :]) ** 2).sum(-1), axis=1)
    for _ in range(max_spins):
        for j in range(k):
            members = vs[assign == j]
            if len(members):
                cen[j] = members.mean(axis=0)
        new = np.argmin(((vs[:, None, :] - cen[None, :, :]) ** 2).sum(-1), axis=1)
        if np.array_equal(new, assign):
            assign = new
            break
        assign = new
    sizes = np.array([max(1, int((assign == j).sum())) for j in range(k)], dtype=float)
    return cen, sizes


# ---------- example -> the two arms' inputs ----------
def plain_input(world, tokens):
    X = world["E"][tokens] + world["Ppos"][:len(tokens)]
    return X, np.ones(len(tokens))


def silo_input(world, tokens, k=K):
    """The CONTENT silo (v3): k-means on token embeddings -> K intents. Orderless."""
    cen, sizes = centrifuge(world["E"][tokens], k)
    return cen, sizes


def position_silo_input(world, tokens, k=K):
    """The POSITION silo: chunk the sequence into K ordered bins; each intent is
    the mean embedding of its bin plus a per-bin positional code, so the encoder
    can tell the bins apart -- this branch KEEPS order. Weighted by bin size."""
    n = len(tokens)
    edges = np.linspace(0, n, k + 1).astype(int)
    intents, sizes = [], []
    for b in range(k):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi <= lo:
            hi = lo + 1
        idx = list(range(lo, min(hi, n)))
        emb = world["E"][[tokens[i] for i in idx]].mean(axis=0)
        intents.append(emb + world["BinPos"][b])
        sizes.append(len(idx))
    return np.stack(intents), np.array(sizes, dtype=float)


# ---------- the two tasks ----------
def _bag(rng):
    return rng.integers(0, G, size=N)                    # a group per position


def _sample(world, task, n_examples, seed):
    rng = np.random.default_rng(seed)
    tg = world["tok_group"]
    tok_of_group = [np.where(tg == g)[0] for g in range(G)]
    X, Y = [], []
    for _ in range(n_examples):
        groups = _bag(rng)
        tokens = np.array([rng.choice(tok_of_group[g]) for g in groups])
        if task == "plurality":
            counts = np.bincount(groups, minlength=G)
            y = int(np.argmax(counts))                   # most frequent group (ties -> lowest)
        elif task == "first":
            y = int(groups[0])                           # the first token's group
        else:
            raise ValueError(task)
        X.append(tokens); Y.append(y)
    return X, np.array(Y)


def dataset(world, task, n_train=800, n_test=400, seed=100):
    Xtr, Ytr = _sample(world, task, n_train, seed)
    Xte, Yte = _sample(world, task, n_test, seed + 1)
    return {"task": task, "Xtr": Xtr, "Ytr": Ytr, "Xte": Xte, "Yte": Yte,
            "n_classes": G, "chance": 1.0 / G}


# ---------- v5: the MIXED-regime dataset (a cue tells you order vs no-order) ----
# Each example is either a BAG (label = plurality of the content groups; order
# does NOT matter) or an ORDER example (label = the first content token's group;
# order DOES matter). A regime cue token is prepended so the router CAN detect the
# regime from the input -- the honest condition for routing to be possible at all.
def _tok_of_group(world):
    tg = world["tok_group"]
    return [np.where(tg == g)[0] for g in range(G)]


def _mixed_sample(world, n_examples, seed):
    rng = np.random.default_rng(seed)
    tg = _tok_of_group(world)
    examples = []
    for _ in range(n_examples):
        r = int(rng.integers(0, 2))                      # 0 = bag, 1 = order
        groups = rng.integers(0, G, size=N)
        tokens = np.array([rng.choice(tg[g]) for g in groups])
        if r == 0:
            y = int(np.argmax(np.bincount(groups, minlength=G)))   # plurality
        else:
            y = int(groups[0])                                     # first token
        examples.append((r, tokens, y))
    return examples


def moe_inputs(world, r, tokens):
    """Return (silo_X, silo_w, plain_X, plain_w) for one example.
    silo = content k-means (orderless, cheap). plain = regime marker + tokens +
    positions (order-aware; the router reads its mean to detect the regime)."""
    cen, sizes = centrifuge(world["E"][tokens], K)                 # content silo
    marker = (world["RegE"][r] + world["MarkPos"])[None, :]
    body = world["E"][tokens] + world["Ppos"][:len(tokens)]
    plain_X = np.vstack([marker, body])                           # (N+1, D)
    return cen, sizes, plain_X, np.ones(plain_X.shape[0])


def mixed_dataset(world, n_train=1000, n_test=500, seed=200):
    tr = _mixed_sample(world, n_train, seed)
    te = _mixed_sample(world, n_test, seed + 1)
    return {"train": tr, "test": te, "n_classes": G, "chance": 1.0 / G}
