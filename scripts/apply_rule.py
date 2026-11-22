#!/usr/bin/env python3
"""apply_rule.py -- the pass rule of results/PREREGISTRATION.md applied to the final rows.

    python3 -I scripts/apply_rule.py [--speed results/final/speed.json] [--json out.json]

Speed from speed_compare.py's --json; quality from the per-position dumps of results/kld/q16-<cache>-<arm> against q16-<cache>-d0
(paired: dKLD = mean of the arm's KL divergence minus delta 0's on the same positions; perplexity ratio = exp of the mean difference
of the two rows' log-likelihood losses).  A delta passes when gain >= +5 %, dKLD <= 0.0045 and the perplexity ratio within 1 % of 1; the pick
is the largest-gain passing delta, a gain within 1 point of a smaller passing delta's going to the smaller one.
"""
import argparse, json, math, pathlib, struct
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
REC = np.dtype([("kld", "<f4"), ("p_diff", "<f4"), ("nll", "<f4"), ("nll_base", "<f4"), ("top", "<i4"), ("top_base", "<i4")])
GAIN_MIN, DKLD_MAX, PPL_MAX, TIE = 0.05, 0.0045, 1.01, 0.01


def load(name):
    raw = (ROOT / "results/kld" / name / "positions.bin").read_bytes()
    n, = struct.unpack_from("<I", raw, 4)
    return np.frombuffer(raw, dtype=REC, count=n, offset=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", default=str(ROOT / "results/final/speed.json"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    speed = json.load(open(a.speed))
    out = {}
    for cache, rep in speed.items():
        base = load(f"q16-{cache}-d0")
        rows = []
        for arm, s in rep["arms"].items():
            q = load(f"q16-{cache}-{arm}")
            dkld = float((q["kld"].astype(np.float64) - base["kld"]).mean())
            ppl = math.exp(float((q["nll"].astype(np.float64) - base["nll"]).mean()))
            delta = float(arm.replace("add", ""))
            ok = {"speed": s["gain_mean"] >= GAIN_MIN, "kld": dkld <= DKLD_MAX, "ppl": abs(ppl - 1) <= PPL_MAX - 1}
            rows.append({"arm": arm, "delta": delta, "gain": s["gain_mean"], "dkld": dkld, "ppl_ratio": ppl, "pass": all(ok.values()),
                         "fails": [k for k, v in ok.items() if not v]})
        rows.sort(key=lambda r: r["delta"])
        passing = [r for r in rows if r["pass"]]
        pick = None
        if passing:
            best = max(passing, key=lambda r: r["gain"])
            pick = min((r for r in passing if r["delta"] <= best["delta"] and r["gain"] >= best["gain"] - TIE), key=lambda r: r["delta"])
        print(f"cache {cache} MiB")
        for r in rows:
            mark = "pass" if r["pass"] else "fails " + ", ".join(r["fails"])
            print(f"  delta {r['delta']:<8g} gain {r['gain']*100:+6.2f} %  dKLD {r['dkld']:+.5f}  PPL ratio {r['ppl_ratio']:.4f}  {mark}")
        print(f"  -> {'delta ' + format(pick['delta'], 'g') if pick else 'delta 0 (none passes)'}")
        out[cache] = {"rows": rows, "pick": pick["delta"] if pick else 0.0}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
