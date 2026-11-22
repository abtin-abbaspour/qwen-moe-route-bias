#!/usr/bin/env python3
"""speed_blocks.py -- the 5-block speed measurement of results/AMENDMENT-1.md: per block, each arm against that block's delta-0 run.

    python3 -I scripts/speed_blocks.py [--prefix spdc] [--json out.json]

Runs are results/runs/<stamp>-<prefix>-<cache>-<arm>-r<block>/.  Block gain = mean over the 24 prompts of (arm decode rate / delta-0 decode
rate of the same block) - 1; an arm's gain is the median of its block gains (the mean of all its runs against the mean of all delta-0
runs, the registered statistic, alongside).
"""
import argparse, glob, json, pathlib, re, statistics
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent


def rates(d):
    j = json.load(open(d / "decode.json"))
    return {k: v["runs"][0]["timings"]["predicted_per_second"] for k, v in j["decode"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="spdc")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    runs = defaultdict(dict)   # cache -> (arm, block) -> dir
    for d in sorted(glob.glob(str(ROOT / f"results/runs/*-{a.prefix}-*"))):
        m = re.search(rf"/(\d{{8}}T\d{{6}}Z)-{a.prefix}-(\d+)-(.+)-r(\d+)$", d)
        if m and (pathlib.Path(d) / "decode.json").exists():
            runs[m.group(2)][(m.group(3), int(m.group(4)))] = pathlib.Path(d)
    report = {}
    for cache in sorted(runs, key=int):
        R = {k: rates(d) for k, d in runs[cache].items()}
        blocks = sorted({b for (arm, b) in R if arm == "d0"})
        prompts = sorted(next(iter(R.values())))
        base_all = {p: statistics.fmean(R[("d0", b)][p] for b in blocks) for p in prompts}
        print(f"cache {cache} MiB: {len(blocks)} blocks; delta-0 run means "
              + ", ".join(f"{statistics.fmean(R[('d0', b)].values()):.1f}" for b in blocks) + " tok/s")
        rep = {}
        for arm in sorted({x for (x, b) in R if x != "d0"}, key=lambda s: (not s.startswith("add"), float(s[3:]) if s.startswith("add") else 0, s)):
            bg_ = [b for b in blocks if (arm, b) in R]
            gains = [statistics.fmean(R[(arm, b)][p] / R[("d0", b)][p] for p in prompts) - 1 for b in bg_]
            pooled = statistics.fmean(statistics.fmean(R[(arm, b)][p] for b in bg_) / base_all[p] for p in prompts) - 1
            med = statistics.median(gains)
            print(f"  {arm:<12} block gains {', '.join(f'{g*100:+.2f}' for g in gains)} -> median {med*100:+.2f} %"
                  f" (pooled mean, registered statistic: {pooled*100:+.2f} %)")
            rep[arm] = {"block_gains": gains, "median_gain": med, "pooled_gain": pooled}
        report[cache] = rep
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
