#!/usr/bin/env python3
"""figures.py -- the README's figures as static SVGs, one light and one dark file each, drawn from the files in results/.

    python3 -I scripts/figures.py        writes docs/<figure>-light.svg and docs/<figure>-dark.svg

  setup       the machine: where the experts live and the two paths a routed expert can take
  bias        how the bias flips a near-tie onto the GPU (an illustrative token, typical Qwen router probabilities)
  pipeline    the experiment from calibration to the GSM8K check
  frontier    decode speed gain against added KL divergence, per cache size (results/final/rule.json, speed-confirm.json)
  layers      quality cost per swapped expert, by layer block (screening rows results/kld/s45-L*)
Stdlib only.
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
FONT = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'
THEMES = {  # data-viz reference palette: categorical slots 1-3 and 7, chart chrome; each mode its own steps
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  wash="rgba(11,11,11,0.045)", series=["#2a78d6", "#eb6834", "#1baf7a"], gpu="#6250d6",
                  gpu_soft="rgba(98,80,214,0.35)", gpu_wash="rgba(98,80,214,0.07)", ram="#b4b3ab", box="#f3f2ee"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 wash="rgba(255,255,255,0.05)", series=["#3987e5", "#d95926", "#199e70"], gpu="#9085e9",
                 gpu_soft="rgba(144,133,233,0.40)", gpu_wash="rgba(144,133,233,0.10)", ram="#62615c", box="#232322"),
}
CACHES = ["3000", "4500", "8000"]


def head(W, H, t, label, title=None, sub=None):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" {FONT} role="img" '
         f'aria-label="{label}">',
         '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{t["ink2"]}"/></marker>'
         '<marker id="arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{t["gpu"]}"/></marker></defs>',
         f'<rect width="{W}" height="{H}" rx="12" fill="{t["surface"]}"/>']
    if title:
        o.append(f'<text x="28" y="36" font-size="19" font-weight="600" fill="{t["ink"]}">{title}</text>')
    if sub:
        o.append(f'<text x="28" y="60" font-size="13" fill="{t["ink2"]}">{sub}</text>')
    return o


def text(x, y, s, t, size=12.5, color="ink2", anchor="start", weight=None):
    w = f' font-weight="{weight}"' if weight else ""
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{t[color]}"{w}>{s}</text>'


def box(x, y, w, h, t, stroke=None, fill=None, rx=10, sw=1.2):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill or t["box"]}" stroke="{stroke or t["axis"]}" stroke-width="{sw}"/>'


def arrow(x1, y1, x2, y2, t, gpu=False, sw=1.5):
    col, m = (t["gpu"], "arrg") if gpu else (t["ink2"], "arr")
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{sw}" marker-end="url(#{m})"/>'


# ---------------------------------------------------------------- setup
def fig_setup(t):
    W, H = 960, 520
    o = head(W, H, t, "The machine: all routed experts live in system RAM; the GPU holds attention, the shared expert and a cache of "
             "routed experts. A routed expert in the cache runs on the GPU; a missed one is computed by the CPU while the GPU waits.",
             "The setup: where the experts live", "One RTX 5070 Ti 16 GB, a Ryzen 9 7950X3D and 32 GB of DDR5. "
             "Qwen3.6-35B-A3B: 40 MoE layers, 256 experts each, 8 chosen per token.")
    # RAM and CPU column
    o.append(box(28, 92, 300, 128, t))
    o.append(text(46, 120, "System RAM, 32 GB", t, 14, "ink", weight=600))
    o.append(text(46, 146, "All routed experts: 40 layers × 256", t))
    o.append(text(46, 166, "15 GB of weights, pinned for the GPU", t))
    o.append(text(46, 196, "Every expert is here, cached or not", t, 12, "muted"))
    o.append(box(28, 268, 300, 112, t))
    o.append(text(46, 296, "CPU, 16 cores", t, 14, "ink", weight=600))
    o.append(text(46, 322, "Computes 90 % of cache misses", t))
    o.append(text(46, 342, "from the weights in RAM", t))
    o.append(text(46, 366, "The GPU waits for the result", t, 12, "muted"))
    o.append(arrow(178, 222, 178, 264, t))
    o.append(text(188, 248, "reads expert weights", t, 12, "muted"))
    # GPU column
    o.append(box(520, 92, 412, 288, t))
    o.append(text(538, 120, "GPU, 16 GB", t, 14, "ink", weight=600))
    o.append(box(538, 136, 376, 74, t, fill=t["surface"]))
    o.append(text(554, 160, "Always on the GPU", t, 12.5, "ink", weight=600))
    o.append(text(554, 182, "attention, router, shared expert, MTP head,", t))
    o.append(text(554, 200, "KV cache (q8_0, 64K context)", t))
    o.append(box(538, 228, 376, 134, t, stroke=t["gpu"], fill=t["gpu_wash"], sw=1.6))
    o.append(text(554, 252, "Expert cache: 3,000 / 4,500 / 8,000 MiB", t, 12.5, "ink", weight=600))
    o.append(text(554, 274, "the routed experts used most often lately", t))
    o.append(text(554, 294, "hit rate 63 / 75 / 90 % without the bias", t))
    o.append(text(554, 322, "The route bias nudges near-ties", t, 12.5, "gpu", weight=600))
    o.append(text(554, 342, "toward experts already in here", t, 12.5, "gpu", weight=600))
    # the two paths
    o.append(arrow(330, 140, 516, 140, t, gpu=True, sw=1.8))
    o.append(text(423, 130, "fills the cache (PCIe Gen4)", t, 12, "gpu", "middle"))
    o.append(arrow(516, 300, 332, 300, t))
    o.append(text(424, 290, "a miss: activations out", t, 12, "muted", "middle"))
    o.append(arrow(332, 340, 516, 340, t))
    o.append(text(424, 360, "the expert's result back", t, 12, "muted", "middle"))
    # what was measured
    y = 404
    o.append(text(28, y + 14, "Measured on every setting", t, 12.5, "ink", weight=600))
    items = [("Decode speed", "24 prompts × 512 tokens, fresh server"), ("Quality", "KL divergence against an f16 reference"),
             ("Accuracy", "GSM8K, all 1,319 questions")]
    for i, (a, b) in enumerate(items):
        x = 28 + i * 304
        o.append(box(x, y + 26, 296, 62, t, fill=t["surface"]))
        o.append(text(x + 16, y + 50, a, t, 12.5, "ink", weight=600))
        o.append(text(x + 16, y + 72, b, t, 12.5))
    o.append("</svg>")
    return o


# ---------------------------------------------------------------- bias
def fig_bias(t):
    W, H = 960, 440
    o = head(W, H, t, "How the route bias works: adding a small delta to the router probability of experts already in the GPU cache "
             "moves a cached expert from 9th to 8th place, so it runs on the GPU instead of an uncached expert running on the CPU.",
             "How the route bias works: one near-tie, before and after",
             "One token in one layer (illustrative; typical Qwen probabilities). Ranks 1 to 5 are unchanged; the top 8 run.")
    rows0 = [("203", 0.0141, True), ("88", 0.0133, False), ("117", 0.0123, False), ("42", 0.0115, True), ("9", 0.0104, False)]
    delta = 0.001
    rows1 = sorted([(e, p, c) for e, p, c in rows0], key=lambda r: -(r[1] + (delta if r[2] else 0)))
    scale = 170 / 0.016

    def panel(x0, rows, biased, title):
        o.append(text(x0, 98, title, t, 14, "ink", weight=600))
        y0, rh = 118, 44
        for i, (e, p, cached) in enumerate(rows):
            y = y0 + i * rh
            rank = 6 + i
            o.append(text(x0, y + 22, f"#{rank}", t, 12.5, "muted"))
            o.append(text(x0 + 34, y + 22, f"expert {e}", t, 12.5, "ink"))
            o.append(text(x0 + 112, y + 22, "GPU cache" if cached else "RAM only", t, 12, "gpu" if cached else "muted", "start",
                          600 if cached else None))
            bx = x0 + 192
            o.append(f'<rect x="{bx}" y="{y + 8}" width="{p * scale:.1f}" height="20" rx="4" fill="{t["gpu"] if cached else t["ram"]}"/>')
            end = bx + p * scale
            if biased and cached:
                o.append(f'<rect x="{end + 2:.1f}" y="{y + 8}" width="{delta * scale - 2:.1f}" height="20" rx="4" fill="{t["gpu_soft"]}"/>')
                o.append(text(f"{end + delta * scale + 8:.1f}", y + 22, f"{p:.4f} + δ", t, 12.5, "gpu", weight=600))
            else:
                o.append(text(f"{end + 8:.1f}", y + 22, f"{p:.4f}", t, 12.5))
            if i == 2:  # the top-8 cut sits under rank 8
                yc = y + rh - 2
                o.append(f'<line x1="{x0}" y1="{yc}" x2="{x0 + 430}" y2="{yc}" stroke="{t["ink2"]}" stroke-width="1.2" stroke-dasharray="5 4"/>')
                o.append(text(x0 + 430, yc - 6, "top-8 cut", t, 11.5, "muted", "end"))
        return y0 + len(rows) * rh

    yb = panel(28, rows0, False, "Without the bias (δ = 0)")
    panel(512, rows1, True, "With the bias (δ = 0.001 on cached experts)")
    o.append(arrow(462, 220, 498, 220, t, sw=1.8))
    y = yb + 30
    o.append(text(28, y, "Expert 117 makes the cut but is in RAM:", t, 12.5, "ink", weight=600))
    o.append(text(28, y + 20, "the CPU computes it while the GPU waits.", t))
    o.append(text(512, y, "Expert 42 is already on the GPU and takes 8th place:", t, 12.5, "ink", weight=600))
    o.append(text(512, y + 20, "no CPU work. It is mixed in at its true weight, 0.0115.", t))
    o.append("</svg>")
    return o


# ---------------------------------------------------------------- pipeline
def fig_pipeline(t):
    W, H = 1040, 270
    o = head(W, H, t, "The experiment in six steps: calibrate, screen bias forms, fix the pass rule, measure, measure the "
             "small-cache speeds in shuffled blocks, and confirm the picks on GSM8K.",
             "The experiment, in order", "Steps 1 and 2 explored; from step 3 on, a rule fixed in advance decided.")
    steps = [("1  Calibrate", ["cache sizes, CPU", "share, noise floors"], False),
             ("2  Screen", ["five bias forms", "at 4,500 MiB"], False),
             ("3  Fix the rule", ["17:10 UTC, before", "any final number"], True),
             ("4  Measure", ["48 speed runs,", "19 quality rows"], True),
             ("5  Block runs", ["small caches: 40", "shuffled starts"], True),
             ("6  Confirm", ["GSM8K, 1,319", "questions per arm"], True)]
    bw, gap, x0, y0, bh = 150, 20, 28, 96, 104
    for i, (title, lines, ruled) in enumerate(steps):
        x = x0 + i * (bw + gap)
        o.append(box(x, y0, bw, bh, t, stroke=t["gpu"] if ruled else t["axis"], fill=t["gpu_wash"] if ruled else t["box"],
                     sw=1.5 if ruled else 1.2))
        o.append(text(x + 14, y0 + 28, title, t, 13.5, "ink", weight=600))
        for k, s in enumerate(lines):
            o.append(text(x + 14, y0 + 54 + k * 20, s, t, 12))
        if i < len(steps) - 1:
            o.append(arrow(x + bw + 2, y0 + bh / 2, x + bw + gap - 2, y0 + bh / 2, t))
    # brackets
    yb = y0 + bh + 22
    xa, xb = x0, x0 + 2 * bw + gap
    o.append(f'<path d="M{xa},{yb - 8} V{yb} H{xb} V{yb - 8}" fill="none" stroke="{t["muted"]}" stroke-width="1.2"/>')
    o.append(text((xa + xb) / 2, yb + 20, "exploratory: chose the bias's form", t, 12, "muted", "middle"))
    xc, xd = x0 + 2 * (bw + gap), x0 + 6 * bw + 5 * gap
    o.append(f'<path d="M{xc},{yb - 8} V{yb} H{xd} V{yb - 8}" fill="none" stroke="{t["gpu"]}" stroke-width="1.2"/>')
    o.append(text((xc + xd) / 2, yb + 20, "decided by the rule fixed in step 3 (one written amendment, in step 5)", t, 12, "gpu", "middle", 600))
    o.append("</svg>")
    return o


# ---------------------------------------------------------------- frontier
def frontier_data():
    rule = json.load(open(ROOT / "results/final/rule.json"))
    conf = json.load(open(ROOT / "results/final/speed-confirm.json"))
    out = {}
    for c in CACHES:
        pts = [(0.0, 0.0, 0.0)]
        for r in rule[c]["rows"]:
            g = conf.get(c, {}).get(r["arm"], {}).get("median_gain", r["gain"])
            pts.append((r["delta"], r["dkld"], g * 100))
        out[c] = pts
    return out


def fig_frontier(t):
    W, H = 920, 540
    L, R, T, B = 78, 150, 118, 70
    xmax, ymin, ymax = 0.022, -2.0, 28.0
    DKLD_MAX, GAIN_MIN = 0.0045, 5.0
    PICKS = {"3000": 0.001, "8000": 0.0025}
    LABEL_OFFSET = {"8000": (8, 24)}  # its line ends on the 3,000 MiB line: label below the end
    X = lambda v: L + (W - L - R) * v / xmax
    Y = lambda v: T + (H - T - B) * (ymax - v) / (ymax - ymin)
    o = head(W, H, t, "Decode speed gain against added KL divergence for 3,000, 4,500 and 8,000 MiB expert caches. Within the quality "
             "limit the bias gives about 5 % at the two small caches and 7 % at 8,000 MiB; larger deltas give up to 25 % at several "
             "times the quality cost.", "Route bias: decode speed gain against quality cost",
             "Qwen3.6-35B-A3B IQ4_XS, RTX 5070 Ti 16 GB. Points, left to right: bias delta 0, 0.00025, 0.0005, 0.001, 0.0025, 0.005.")
    o[-2] = o[-2].replace('x="28"', f'x="{L}"'); o[-1] = o[-1].replace('x="28"', f'x="{L}"')
    lx = L
    for i, c in enumerate(CACHES):
        col = t["series"][i]
        o.append(f'<line x1="{lx}" y1="86" x2="{lx + 22}" y2="86" stroke="{col}" stroke-width="2" stroke-linecap="round"/>'
                 f'<circle cx="{lx + 11}" cy="86" r="4.5" fill="{col}" stroke="{t["surface"]}" stroke-width="2"/>'
                 + text(lx + 30, 90.5, f"{int(c):,} MiB cache", t, 13))
        lx += 150
    o.append(f'<circle cx="{lx + 11}" cy="86" r="8" fill="none" stroke="{t["ink2"]}" stroke-width="1.5"/>'
             + text(lx + 26, 90.5, "picked by the pre-registered rule", t, 13))
    o.append(f'<rect x="{X(0)}" y="{Y(ymax)}" width="{X(DKLD_MAX) - X(0)}" height="{Y(GAIN_MIN) - Y(ymax)}" fill="{t["wash"]}"/>')
    for v in range(0, 30, 5):
        o.append(f'<line x1="{L}" y1="{Y(v)}" x2="{W - R}" y2="{Y(v)}" stroke="{t["grid"]}" stroke-width="1"/>'
                 + text(L - 10, Y(v) + 4.5, f"+{v} %", t, 12, "muted", "end"))
    for v in (0, 0.005, 0.010, 0.015, 0.020):
        o.append(text(X(v), H - B + 22, f"{v:.3f}", t, 12, "muted", "middle"))
    o.append(f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="{t["axis"]}" stroke-width="1"/>')
    o.append(text((L + W - R) / 2, H - 18, "Added KL divergence against delta 0 (lower is better)", t, 13, "ink2", "middle"))
    o.append(f'<text transform="translate(22 {(T + H - B) / 2}) rotate(-90)" font-size="13" text-anchor="middle" fill="{t["ink2"]}">'
             'Decode speed gain against delta 0</text>')
    o.append(f'<line x1="{X(DKLD_MAX)}" y1="{Y(ymax)}" x2="{X(DKLD_MAX)}" y2="{H - B}" stroke="{t["ink2"]}" stroke-width="1"/>'
             + text(X(DKLD_MAX) + 8, Y(ymax) + 14, "Quality limit 0.0045:", t, 12)
             + text(X(DKLD_MAX) + 8, Y(ymax) + 30, "what the 8-bit KV cache already costs", t, 12))
    o.append(f'<line x1="{L}" y1="{Y(GAIN_MIN)}" x2="{W - R}" y2="{Y(GAIN_MIN)}" stroke="{t["ink2"]}" stroke-width="1"/>'
             + text(W - R, Y(GAIN_MIN) - 7, "Speed limit +5 %", t, 12, "ink2", "end"))
    o.append(text(X(0) + 8, Y(ymax) + 14, "passes both", t, 12, "muted"))
    data = frontier_data()
    for i, c in enumerate(CACHES):
        col = t["series"][i]
        pts = data[c]
        path = " ".join(f"{'M' if k == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for k, (_, x, y) in enumerate(pts))
        o.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for d, x, y in pts:
            if PICKS.get(c) == d:
                o.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="9" fill="none" stroke="{t["ink2"]}" stroke-width="1.5"/>')
            o.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4.5" fill="{col}" stroke="{t["surface"]}" stroke-width="2"/>')
        d, x, y = pts[-1]
        dx, dy = LABEL_OFFSET.get(c, (12, 4.5))
        o.append(text(f"{X(x) + dx:.1f}", f"{Y(y) + dy:.1f}", f"{int(c):,} MiB", t, 13, "ink", weight=600))
    o.append("</svg>")
    return o


# ---------------------------------------------------------------- layers
def fig_layers(t):
    J = lambda n: json.load(open(ROOT / f"results/kld/{n}/summary.json"))
    floor = J("s45-mul1e-30")["kld_mean"]   # a perturbed run with no swaps: the noise floor every biased row sits on
    blocks = [("0–7", "s45-L00-07-0.001"), ("8–15", "s45-L08-15-0.001"), ("16–23", "s45-L16-23-0.001"),
              ("24–31", "s45-L24-31-0.001"), ("32–39", "s45-L32-39-0.001")]
    vals = []
    for lab, n in blocks:
        s = J(n)
        vals.append((lab, (s["kld_mean"] - floor) / (s["swap_rate_picks"] * 100)))
    W, H = 920, 400
    L, R, T, B = 96, 40, 96, 70
    ymax = 0.003
    Y = lambda v: T + (H - T - B) * (ymax - v) / ymax
    o = head(W, H, t, "Quality cost per swapped expert by layer block: the first 16 layers cost several times more KL divergence per "
             "swap than the last 24.", "Where the quality cost comes from: early layers",
             "Bias 0.001 in one block of 8 layers at a time (4,500 MiB, screening precision). Added KL divergence per 1 % of picks swapped.")
    o[-2] = o[-2].replace('x="28"', f'x="{L}"'); o[-1] = o[-1].replace('x="28"', f'x="{L}"')
    for v in (0, 0.001, 0.002, 0.003):
        o.append(f'<line x1="{L}" y1="{Y(v)}" x2="{W - R}" y2="{Y(v)}" stroke="{t["grid"]}" stroke-width="1"/>'
                 + text(L - 10, Y(v) + 4.5, f"{v:.3f}", t, 12, "muted", "end"))
    o.append(f'<line x1="{L}" y1="{Y(0)}" x2="{W - R}" y2="{Y(0)}" stroke="{t["axis"]}" stroke-width="1"/>')
    slot = (W - L - R) / len(vals)
    col = t["series"][0]
    for i, (lab, v) in enumerate(vals):
        cx = L + slot * (i + 0.5)
        h = Y(0) - Y(v)
        o.append(f'<path d="M{cx - 12},{Y(0)} V{Y(v) + 4} Q{cx - 12},{Y(v)} {cx - 8},{Y(v)} H{cx + 8} Q{cx + 12},{Y(v)} {cx + 12},{Y(v) + 4} V{Y(0)} Z" fill="{col}"/>')
        o.append(text(cx, Y(0) + 22, f"layers {lab}", t, 12.5, "ink2", "middle"))
    o.append(f'<text transform="translate(26 {(T + H - B) / 2}) rotate(-90)" font-size="13" text-anchor="middle" fill="{t["ink2"]}">'
             'added KL divergence per 1 % swapped</text>')
    hi, lo = max(v for _, v in vals[:2]), min(v for _, v in vals[2:])
    early, late = [v for _, v in vals[:2]], [v for _, v in vals[2:]]
    o.append(text(W - R, T + 4, f"per swap, layers 0–15 cost {min(early) / max(late):.1f}× to {max(early) / min(late):.0f}× as much as layers 16–39",
                  t, 12, "muted", "end"))
    o.append(text(L, H - 22, "Biasing only late layers gave about the same cost per point of hit rate (it needs a larger δ "
             "in fewer layers), so the final runs bias all 40.", t, 12, "muted"))
    o.append("</svg>")
    return o


FIGS = {"setup": fig_setup, "bias": fig_bias, "pipeline": fig_pipeline, "frontier": fig_frontier, "layers": fig_layers}


def main():
    (ROOT / "docs").mkdir(exist_ok=True)
    for name, f in FIGS.items():
        for theme, t in THEMES.items():
            (ROOT / f"docs/{name}-{theme}.svg").write_text("\n".join(f(t)) + "\n")
    print("wrote", ", ".join(f"docs/{n}-{{light,dark}}.svg" for n in FIGS))


if __name__ == "__main__":
    main()
