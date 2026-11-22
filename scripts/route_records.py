#!/usr/bin/env python3
"""route_records.py -- read the per-token route records (LLAMA_MOE_ROUTE_STATS=<path>, "MRS1") and replay the router offline.

    python3 -I scripts/route_records.py <file.mrs> [--deltas 0.001,0.002,...] [--mode add|mul|both] [--protect R] [--target 0.5,1,2,4]

For each record (one token at one layer, one decode step) the file holds the experts ranked by the unbiased router score (the top 24,
plus any selected expert below them), their scores, and whether the bias favoured them at that step (cached or being filled) and
whether they were selected.  Replaying a bias delta on those scores, with the favoured set held as it was, gives the swaps that delta
would make at that step: a static, first-order estimate (it ignores that the bias changes what the cache holds next).  From a run
at delta ~ 0 this maps delta to swap rate for both bias rules without running the model; --target solves for the delta of each
swap rate (percent of picks).  "truncated" counts records where an expert below the listed ones could also have entered.
"""
import argparse, struct, sys
import numpy as np


def load(path):
    raw = open(path, "rb").read()
    if raw[:4] != b"MRS1":
        sys.exit(f"{path}: not an MRS1 file")
    k, n_exp = struct.unpack_from("<II", raw, 4)
    bias, = struct.unpack_from("<f", raw, 12)
    mode, protect = struct.unpack_from("<II", raw, 16)
    off, hdr = 24, []
    while off + 8 <= len(raw):
        step, il, t, n = struct.unpack_from("<IHBB", raw, off)
        hdr.append((off + 8, step, il, t, n))
        off += 8 + 8 * n
    W = max(h[4] for h in hdr)
    R = len(hdr)
    ent = np.dtype([("e", "<u2"), ("flags", "u1"), ("pad", "u1"), ("p", "<f4")])
    p = np.full((R, W), -np.inf, dtype=np.float64)
    fl = np.zeros((R, W), dtype=np.uint8)
    for i, (o, step, il, t, n) in enumerate(hdr):
        a = np.frombuffer(raw, dtype=ent, count=n, offset=o)
        p[i, :n] = a["p"]
        fl[i, :n] = a["flags"]
    meta = np.array([(h[1], h[2], h[3], h[4]) for h in hdr], dtype=np.int64)
    return dict(k=k, n_expert=n_exp, bias=bias, mode="mul" if mode else "add", protect=protect, p=p, flags=fl, meta=meta)


def replay(d, delta, mode, protect=0):
    p, fl, k = d["p"], d["flags"], d["k"]
    fav = (fl & 1).astype(bool)
    valid = np.isfinite(p)
    # the unbiased order is the listing order (records are written by unbiased score, best first)
    rank = np.broadcast_to(np.arange(p.shape[1]), p.shape)
    ref = valid & (rank < k)
    key = np.where(fav, p + delta, p) if mode == "add" else np.where(fav, p * np.exp(delta), p)
    key = np.where(valid, key, -np.inf)
    if protect > 0:
        key = np.where(rank < protect, np.inf, key)
    top = np.argsort(-key, axis=1, kind="stable")[:, :k]
    sel = np.zeros_like(valid)
    np.put_along_axis(sel, top, True, axis=1)
    swaps = k - (sel & ref).sum(1)
    favsel = (sel & fav).sum(1)
    favref = (ref & fav).sum(1)
    psafe = np.where(valid, p, 0.0)
    mass = (psafe * sel).sum() / max((psafe * ref).sum(), 1e-30)  # a ratio of sums, as the engine's log
    # could an unlisted expert (score at most the last listed one's) have entered? only if favoured and its biased score beats the k-th
    n = d["meta"][:, 3]
    last = p[np.arange(len(p)), n - 1]
    kth = np.take_along_axis(key, top[:, k - 1:k], axis=1)[:, 0]
    last_b = last + delta if mode == "add" else last * np.exp(delta)
    trunc = (n >= 24) & (last_b > kth)
    return dict(swap=swaps.mean() / k, tok_swapped=(swaps > 0).mean(), fav_sel=favsel.mean() / k, fav_ref=favref.mean() / k,
                mass=mass, trunc=trunc.mean())


def solve(d, target, mode, protect, lo, hi):
    for _ in range(40):
        mid = (lo * hi) ** 0.5
        if replay(d, mid, mode, protect)["swap"] < target:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--deltas", default=None)
    ap.add_argument("--mode", default="both")
    ap.add_argument("--protect", type=int, default=0)
    ap.add_argument("--target", default=None, help="swap rates in percent of picks, e.g. 0.5,1,2,4")
    a = ap.parse_args()
    d = load(a.file)
    print(f"{a.file}: {len(d['p'])} token-layers, k={d['k']}, n_expert={d['n_expert']}, recorded at bias {d['bias']:g} ({d['mode']}), "
          f"protect {d['protect']}")
    obs = replay(d, 0.0, d["mode"])
    print(f"  as recorded: favoured share of the selection {obs['fav_sel']:.4f}")
    modes = ["add", "mul"] if a.mode == "both" else [a.mode]
    defaults = {"add": [0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.03],
                "mul": [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]}
    for m in modes:
        ds = [float(x) for x in a.deltas.split(",")] if a.deltas else defaults[m]
        print(f"  {m}: delta     swaps/picks  tokens-swapped  favoured-picks (unbiased)  mass-kept  truncated")
        for dl in ds:
            r = replay(d, dl, m, a.protect)
            print(f"  {m}: {dl:<9g} {r['swap']*100:9.3f} %  {r['tok_swapped']*100:12.2f} %  {r['fav_sel']*100:10.2f} % ({r['fav_ref']*100:.2f} %)"
                  f"  {r['mass']*100:8.4f} %  {r['trunc']*100:.3f} %")
        if a.target:
            hi = 1.0 if m == "add" else 10.0
            for t in [float(x) / 100 for x in a.target.split(",")]:
                dl = solve(d, t, m, a.protect, 1e-6, hi)
                print(f"  {m}: swap rate {t*100:g} % of picks at delta ~ {dl:.5g}")


if __name__ == "__main__":
    main()
