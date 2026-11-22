#!/usr/bin/env python3
"""summarize_kld.py -- one line (and summary.json) per kld.sh output directory.

    python3 -I scripts/summarize_kld.py results/kld/<name> [more ...]

Reads kld.log (llama-perplexity's KL-divergence report, the MoE cache's counters and route stats) and cmd.txt (settings, elapsed).
"""
import json, re, sys, pathlib

PATS = {
    "ppl_q":       r"Mean PPL\(Q\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
    "ppl_base":    r"Mean PPL\(base\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
    "ppl_ratio":   r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
    "kld_mean":    r"Mean\s+KLD:\s*([\d.]+)\s*±\s*([\d.]+)",
    "kld_p99":     r"99\.0%\s+KLD:\s*([\d.]+)",
    "same_top":    r"Same top p:\s*([\d.]+)\s*±\s*([\d.]+)\s*%",
    "rms_dp":      r"RMS Δp\s*:\s*([\d.]+)\s*±\s*([\d.]+)\s*%",
    "ppl_final":   r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)",
}
RE_SMALL = re.compile(r"ubatch <= 8: hits = (\d+), misses = (\d+) \((\d+) on the CPU\), hit rate = ([\d.]+)%")
RE_ROUTE = re.compile(r"route total: (\d+) token-layers, ([\d.]+)% with a swap, ([\d.]+) swaps per token-layer \(([\d.]+)% of picks\), "
                      r"unbiased mass kept ([\d.]+)%")


def summarize(d: pathlib.Path) -> dict:
    s = {"dir": d.name}
    cmd = (d / "cmd.txt").read_text() if (d / "cmd.txt").exists() else ""
    m = re.search(r"^settings: (.*)$", cmd, re.M); s["settings"] = m.group(1) if m else None
    m = re.search(r"^elapsed_s: (\d+)", cmd, re.M); s["elapsed_s"] = int(m.group(1)) if m else None
    log = (d / "kld.log").read_text(errors="replace") if (d / "kld.log").exists() else ""
    for key, pat in PATS.items():
        m = re.findall(pat, log)
        if m:
            v = m[-1]
            if isinstance(v, tuple):
                s[key] = float(v[0]); s[key + "_err"] = float(v[1]) if len(v) > 1 else None
            else:
                s[key] = float(v)
    m = RE_SMALL.findall(log)
    if m:
        s["hit_rate"] = float(m[-1][3]) / 100
    m = RE_ROUTE.findall(log)
    if m:
        s["swap_rate_picks"] = float(m[-1][3]) / 100; s["mass_kept"] = float(m[-1][4]) / 100
    json.dump(s, open(d / "summary.json", "w"), indent=1)
    return s


def line(s):
    f = lambda k, fmt: (fmt % s[k]) if s.get(k) is not None else "-"
    return (f"{s['dir']:<30} same-top {f('same_top', '%.3f')} ± {f('same_top_err', '%.3f')} %, KLD {f('kld_mean', '%.5f')}, "
            f"PPL ratio {f('ppl_ratio', '%.4f')}, hit {f('hit_rate', '%.4f')}, swaps {f('swap_rate_picks', '%.4f')}, "
            f"{f('elapsed_s', '%d')} s")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(line(summarize(pathlib.Path(p))))
