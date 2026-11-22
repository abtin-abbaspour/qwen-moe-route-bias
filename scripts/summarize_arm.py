#!/usr/bin/env python3
"""summarize_arm.py -- one line (and summary.json) per measured arm directory written by run_arm.sh.

    python3 -I scripts/summarize_arm.py results/runs/<stamp>-<name> [more dirs ...]

Reads decode.json (bench/probe.py), server.log (the MoE cache's final counters, printed at shutdown) and gpu.csv (nvidia-smi, 10 Hz).
Speed is the server's own decode rate (timings.predicted_per_second) per prompt, summarized as median and mean over the 24 prompts;
the client-side decode rate (probe.py's decode_tps) is kept alongside.
"""
import json, re, statistics, sys, csv, pathlib, hashlib

RE_SMALL = re.compile(r"ubatch <= 8: hits = (\d+), misses = (\d+) \((\d+) on the CPU\), hit rate = ([\d.]+)%, uploaded = ([\d.]+) MiB")
RE_ROUTE = re.compile(r"route total: (\d+) token-layers, ([\d.]+)% with a swap, ([\d.]+) swaps per token-layer \(([\d.]+)% of picks\), "
                      r"unbiased mass kept ([\d.]+)%, favoured picks ([\d.]+)% \(unbiased top-k ([\d.]+)%\), swapped-in gap mean ([\d.]+) max ([\d.]+)")
RE_LAYER = re.compile(r"route swaps per token by layer:(.*)")
RE_SLOTS = re.compile(r"llama_moe_cache: .*?(\d+) slots")


def summarize(d: pathlib.Path) -> dict:
    s = {"dir": d.name}
    man = (d / "manifest.txt").read_text() if (d / "manifest.txt").exists() else ""
    for key in ("name", "args", "gpu_mem_used_mib_loaded"):
        m = re.search(rf"^{key}: (.*)$", man, re.M)
        s[key] = m.group(1).strip() if m else None

    dj = json.load(open(d / "decode.json")) if (d / "decode.json").exists() else None
    if dj:
        eng, cli, dn, da, toks, texts = [], [], 0, 0, 0, {}
        for name, v in dj["decode"].items():
            r = v["runs"][0]
            tm = r.get("timings") or {}
            if tm.get("predicted_per_second"):
                eng.append(tm["predicted_per_second"])
            if r.get("decode_tps"):
                cli.append(r["decode_tps"])
            dn += tm.get("draft_n") or 0
            da += tm.get("draft_n_accepted") or 0
            toks += r.get("completion_tokens") or 0
            texts[name] = hashlib.md5(r.get("text", "").encode()).hexdigest()[:10]
        s.update(n_prompts=len(eng), tokens=toks,
                 engine_tps_median=statistics.median(eng) if eng else None, engine_tps_mean=statistics.fmean(eng) if eng else None,
                 client_tps_median=statistics.median(cli) if cli else None,
                 draft_accept=da / dn if dn else None, text_md5=texts)

    log = (d / "server.log").read_text(errors="replace") if (d / "server.log").exists() else ""
    m = RE_SMALL.findall(log)
    if m:
        h, mi, cpu, rate, up = m[-1]
        s.update(hits=int(h), misses=int(mi), hit_rate=float(rate) / 100, uploaded_mib=float(up))
    m = RE_ROUTE.findall(log)
    if m:
        v = m[-1]
        s.update(route_token_layers=int(v[0]), route_tok_swapped=float(v[1]) / 100, swaps_per_token_layer=float(v[2]),
                 swap_rate_picks=float(v[3]) / 100, mass_kept=float(v[4]) / 100, fav_sel=float(v[5]) / 100, fav_ref=float(v[6]) / 100,
                 gap_mean=float(v[7]), gap_max=float(v[8]))
    m = RE_LAYER.findall(log)
    if m:
        s["swaps_by_layer"] = [float(x) for x in m[-1].split()]

    if (d / "gpu.csv").exists():
        pw = []
        with open(d / "gpu.csv") as f:
            for row in csv.reader(f):
                try:
                    pw.append(float(row[1].strip().split()[0]))
                except (ValueError, IndexError):
                    pass
        if pw:
            s["gpu_power_w_mean"] = statistics.fmean(pw)
    json.dump(s, open(d / "summary.json", "w"), indent=1)
    return s


def line(s: dict) -> str:
    f = lambda k, fmt: (fmt % s[k]) if s.get(k) is not None else "-"
    out = (f"{s.get('name') or s['dir']:<28} engine {f('engine_tps_median', '%6.1f')} tok/s (mean {f('engine_tps_mean', '%6.1f')}), "
           f"hit {f('hit_rate', '%.4f')}, draft {f('draft_accept', '%.3f')}, vram {s.get('gpu_mem_used_mib_loaded') or '-'} MiB, "
           f"gpu {f('gpu_power_w_mean', '%.0f')} W")
    if s.get("swap_rate_picks") is not None:
        out += f", swaps {s['swap_rate_picks']:.4f} of picks, mass kept {s['mass_kept']:.4f}"
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(line(summarize(pathlib.Path(p))))
