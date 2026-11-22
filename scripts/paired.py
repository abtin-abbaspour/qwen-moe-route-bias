#!/usr/bin/env python3
"""paired.py -- paired comparisons of kld.sh rows scored against the same base, from their per-position dumps (positions.bin).

    python3 -I scripts/paired.py results/kld/<row> [more rows ...]            one line per row: means with block-bootstrap intervals
    python3 -I scripts/paired.py --vs results/kld/<A> results/kld/<B> [...]    each row against row A: the paired difference

Same top token compares each row's first choice with the base's at each position; two rows differ where one keeps the base's choice
and the other does not, and McNemar's exact test asks whether those discordant positions split evenly.  Intervals come from a block
bootstrap (blocks of 256 consecutive positions, so neighbouring positions are not treated as independent), 2,000 resamples, 95 %.
"""
import argparse, math, pathlib, struct, sys
import numpy as np

REC = np.dtype([("kld", "<f4"), ("p_diff", "<f4"), ("nll", "<f4"), ("nll_base", "<f4"), ("top", "<i4"), ("top_base", "<i4")])
BLOCK, NBOOT = 256, 2000


def load(d):
    raw = (pathlib.Path(d) / "positions.bin").read_bytes()
    if raw[:4] != b"KLD1":
        sys.exit(f"{d}: not a KLD1 dump")
    n, = struct.unpack_from("<I", raw, 4)
    return np.frombuffer(raw, dtype=REC, count=n, offset=8)


def boot(x, rng):
    """block-bootstrap 95 % interval of the mean of x"""
    nb = len(x) // BLOCK
    blocks = x[: nb * BLOCK].reshape(nb, BLOCK).mean(1)
    idx = rng.integers(0, nb, size=(NBOOT, nb))
    means = blocks[idx].mean(1)
    return np.percentile(means, [2.5, 97.5])


def mcnemar(b, c):
    """exact two-sided p-value for b vs c discordant pairs"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vs", action="store_true")
    ap.add_argument("rows", nargs="+")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    data = {r: load(r) for r in a.rows}
    n = {len(v) for v in data.values()}
    if len(n) != 1:
        sys.exit(f"rows differ in length: {n}")
    if not a.vs:
        for r, v in data.items():
            same = (v["top"] == v["top_base"]).astype(np.float64)
            lo, hi = boot(v["kld"].astype(np.float64), rng)
            slo, shi = boot(same, rng)
            print(f"{pathlib.Path(r).name:<26} KLD {v['kld'].mean():.5f} [{lo:.5f}, {hi:.5f}]  same top {same.mean()*100:.3f} % "
                  f"[{slo*100:.3f}, {shi*100:.3f}]  ln PPL ratio {np.mean(v['nll'] - v['nll_base']):+.5f}")
        return
    ref_name = a.rows[0]
    A = data[ref_name]
    sa = A["top"] == A["top_base"]
    print(f"against {pathlib.Path(ref_name).name}: positive = the row is worse")
    for r in a.rows[1:]:
        B = data[r]
        sb = B["top"] == B["top_base"]
        b = int((sa & ~sb).sum()); c = int((~sa & sb).sum())
        dk = (B["kld"] - A["kld"]).astype(np.float64)
        lo, hi = boot(dk, rng)
        ds = (sa.astype(np.float64) - sb.astype(np.float64))
        slo, shi = boot(ds, rng)
        print(f"  {pathlib.Path(r).name:<26} dKLD {dk.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]  same-top drop {ds.mean()*100:+.3f} points "
              f"[{slo*100:+.3f}, {shi*100:+.3f}]  McNemar {b} vs {c}, p = {mcnemar(b, c):.3g}")


if __name__ == "__main__":
    main()
