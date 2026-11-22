#!/usr/bin/env python3
"""speed_compare.py -- decode gain of each biased arm over the interleaved delta-0 runs of the same cache size.

    python3 -I scripts/speed_compare.py [--prefix spd] [--json out.json]

Arms are results/runs/<stamp>-spd-<cache>-<arm>-<rep>/ (scripts/run_arm.sh); the delta-0 runs are spd-<cache>-d0-*.  Per prompt, the
server's decode rate (timings.predicted_per_second) of an arm (the mean of its passes) is divided by the mean of the delta-0 runs
of that prompt; the gain is the mean of the 24 per-prompt ratios minus 1 (as glm53-offload RESULTS.md), with the median alongside.
Noise: each delta-0 run read as an arm against the others (leave-one-out), and the two passes of each arm against each other.
"""
import argparse, glob, json, re, statistics, pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent


def per_prompt(d):
    j = json.load(open(d / "decode.json"))
    out = {}
    for name, v in j["decode"].items():
        tm = v["runs"][0].get("timings") or {}
        out[name] = (tm.get("predicted_per_second"), tm.get("draft_n") or 0, tm.get("draft_n_accepted") or 0,
                     v["runs"][0].get("completion_tokens") or 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="spd")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    runs = defaultdict(lambda: defaultdict(list))   # cache -> arm -> [run dirs]
    for d in sorted(glob.glob(str(ROOT / f"results/runs/*-{a.prefix}-*"))):
        m = re.search(rf"-{a.prefix}-(\d+)-(.+)-(r\d)(\d*)$", d)
        if not m or not (pathlib.Path(d) / "decode.json").exists():
            continue
        runs[m.group(1)][m.group(2)].append(pathlib.Path(d))
    report = {}
    for cache in sorted(runs, key=int):
        arms = runs[cache]
        base_runs = [per_prompt(d) for d in arms.get("d0", [])]
        if not base_runs:
            continue
        prompts = sorted(base_runs[0])
        base = {p: statistics.fmean(r[p][0] for r in base_runs) for p in prompts}
        loo = []
        for i, r in enumerate(base_runs):   # leave-one-out: how far one delta-0 run reads from the others
            others = [x for j, x in enumerate(base_runs) if j != i]
            loo.append(statistics.fmean(r[p][0] / statistics.fmean(o[p][0] for o in others) for p in prompts) - 1)
        print(f"cache {cache} MiB: {len(base_runs)} delta-0 runs, mean decode {statistics.fmean(base.values()):.1f} tok/s; "
              f"one delta-0 run against the rest: {min(loo)*100:+.2f} to {max(loo)*100:+.2f} %")
        rep = {"base_tps": statistics.fmean(base.values()), "loo_min": min(loo), "loo_max": max(loo), "arms": {}}
        for arm in sorted((x for x in arms if x != "d0"), key=lambda s: float(re.sub(r"[^\d.e-]", "", s) or 0)):
            rr = [per_prompt(d) for d in arms[arm]]
            ratios = [statistics.fmean(r[p][0] for r in rr) / base[p] for p in prompts]
            passes = [statistics.fmean(r[p][0] / base[p] for p in prompts) - 1 for r in rr]
            dn = sum(r[p][1] for r in rr for p in prompts); da = sum(r[p][2] for r in rr for p in prompts)
            hit = None
            s = arms[arm][0] / "summary.json"
            if s.exists():
                hit = json.load(open(s)).get("hit_rate")
            g_mean = statistics.fmean(ratios) - 1
            g_med = statistics.median(ratios) - 1
            print(f"  {arm:<14} gain {g_mean*100:+6.2f} % (median {g_med*100:+6.2f} %; passes {', '.join(f'{x*100:+.2f}' for x in passes)}), "
                  f"hit {hit if hit is None else f'{hit:.4f}'}, draft acceptance {da/dn if dn else 0:.3f}")
            rep["arms"][arm] = {"gain_mean": g_mean, "gain_median": g_med, "passes": passes, "hit_rate": hit,
                                "draft_accept": da / dn if dn else None}
        report[cache] = rep
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
