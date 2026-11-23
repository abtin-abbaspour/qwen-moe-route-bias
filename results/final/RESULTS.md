# Route bias on Qwen3.6-35B-A3B: speed against quality on a 16 GB card

Measured 2026-11-18 on one machine:
- **GPU:** RTX 5070 Ti 16 GB, PCIe Gen4 x16, NVIDIA driver 620.70.06; the desktop runs on the same card.
- **CPU and memory:** Ryzen 9 7950X3D, 32 GB DDR5-5200.
- **Model:** Qwen3.6-35B-A3B at IQ4_XS: 256 experts, top-8 plus a shared expert, softmax router, 40 MoE layers, MTP.
- **Engine:** llama.cpp d02b4c0d with the glm53-offload patch, expert cache on the GPU, 90 % of the misses computed on the CPU, q8_0 KV,
  64K context, MTP with 2 drafts.

**The bias.** A constant delta is added to the router's softmax probability of each expert that is in the GPU cache, before the
top-8 choice. A near-tie then goes to an expert the GPU already holds. The mixing weights stay unbiased. The bias is additive and the
same in all 40 layers; calibration and screening (`results/calib/CALIBRATION.md`) found nothing better on this model:
- a logit-space (multiplicative) bias breaks the fused CUDA kernel and costs about 5 % before any swap;
- dropping the displaced expert, protecting the top 2, or restricting or weighting layers all gave the same or a worse frontier.

**The pass rule** was fixed before the final runs: `results/PREREGISTRATION.md`, 17:10 UTC, sha256 f1ec8923…63aa. Speed at 3,000 and
4,500 MiB is the median of 5 shuffled blocks per setting, under `results/AMENDMENT-1.md` (19:50 UTC, sha256 2356d037…ba43), which
was fixed before those runs.

## Result

A delta passes when it meets all three: decode at least 5 % faster than delta 0, added KL divergence (dKLD) at most 0.0045, and
perplexity within 1 %. 0.0045 is what the q8_0 KV cache already costs this model against f16.

| GPU expert cache | Delta 0 decode (median run) | Pick | Decode gain | dKLD | GSM8K, delta 0 → pick |
|---|---|---|---|---|---|
| 3,000 MiB (63 % hit) | 145 tok/s | **0.001** | +5.3 % (median of 5 blocks) | +0.0040 | 94.77 % → 94.39 % (−0.38 points [−1.21, +0.45], p = 0.47): kept |
| 4,500 MiB (75 % hit) | 171 tok/s | **0** (none passes) | 0.001 reaches +4.8 % | — | — |
| 8,000 MiB (90 % hit) | 220 tok/s | **0.0025** | +6.9 % | +0.0041 | 95.07 % → 94.54 % (−0.53 points [−1.29, +0.23], p = 0.23): kept |

Delta 0.001 gives about +5 % at both 3,000 and 4,500 MiB (+5.3 % and +4.8 %). One clearing the 5 % line and the other not is within
the measurement's precision, not a real difference between the cache sizes. In short: within a quality budget equal to the 8-bit KV
cache's cost, the bias buys about 5 % decode at small caches and about 7 % at 8,000 MiB.

Under the speed measurement first registered (two runs per arm), the rule picked 0.0005 at 4,500 MiB (+7.0 %). The block measurement
gives that delta +2.6 %, so no delta passes there.

GSM8K is a different workload (thinking off, short math answers). The picked deltas decoded faster there than on the decode benchmark,
with about the same number of tokens. These are single runs.

| Cache | Delta | Decode, delta 0 → pick | Gain | Tokens, delta 0 / pick |
|---|---|---|---|---|
| 3,000 MiB | 0.001 | 146.1 → 159.6 tok/s | +9.2 % | 460,439 / 460,224 |
| 8,000 MiB | 0.0025 | 244.5 → 270.7 tok/s | +10.7 % | 457,985 / 457,136 |

## The whole curve

Decode gain is from the final sweep: two runs per arm against six delta-0 runs, 24 prompts, 512 tokens, thinking on. The median of 5
shuffled blocks is in brackets where it was measured. Hit rate and draft acceptance come from the speed runs. Swap rate, dKLD,
same-top drop and the perplexity ratio come from the quality runs: wikitext-2, 16 × 4,096-token chunks, 32,752 scored positions,
3-token steps, each row paired with delta 0 on the same positions against an f16-KV reference with every expert on the GPU.
Intervals are 95 % block bootstraps.

**3,000 MiB** (delta 0: hit 62.7 %)

| Delta | Decode gain | Hit | Swaps | dKLD | Same-top drop (points) | PPL ratio | Rule |
|---|---|---|---|---|---|---|---|
| 0.00025 | −1.3 % (+0.8 %) | 63.2 % | 1.0 % | +0.0009 [0.0006, 0.0012] | 0.25 [0.07, 0.42] | 1.0001 | slow |
| 0.0005 | +5.8 % (+2.0 %) | 64.4 % | 2.0 % | +0.0018 [0.0015, 0.0022] | 0.57 [0.35, 0.74] | 0.9987 | slow |
| 0.001 | +8.9 % (+5.3 %) | 65.5 % | 3.8 % | +0.0040 [0.0036, 0.0046] | 1.15 [0.92, 1.37] | 1.0002 | **pass** |
| 0.0025 | +11.0 % | 70.3 % | 8.6 % | +0.0099 [0.0091, 0.0110] | 2.13 [1.86, 2.41] | 1.0023 | KLD |
| 0.005 | +21.0 % | 75.5 % | 14.7 % | +0.0207 [0.0184, 0.0237] | 3.70 [3.39, 4.02] | 1.0062 | KLD |

**4,500 MiB** (delta 0: hit 75.1 %)

| Delta | Decode gain | Hit | Swaps | dKLD | Same-top drop (points) | PPL ratio | Rule |
|---|---|---|---|---|---|---|---|
| 0.00025 | +3.3 % (+0.3 %) | 76.4 % | 0.9 % | +0.0008 [0.0005, 0.0011] | 0.18 [0.00, 0.38] | 0.9990 | slow |
| 0.0005 | +7.0 % (+2.6 %) | 76.8 % | 1.7 % | +0.0021 [0.0017, 0.0025] | 0.51 [0.32, 0.72] | 0.9993 | slow |
| 0.001 | +5.8 % (+4.8 %) | 78.6 % | 3.2 % | +0.0037 [0.0033, 0.0041] | 0.97 [0.75, 1.18] | 1.0000 | slow |
| 0.0025 | +12.3 % | 82.7 % | 6.8 % | +0.0084 [0.0074, 0.0096] | 1.86 [1.63, 2.09] | 1.0028 | KLD |
| 0.005 | +25.4 % | 87.0 % | 11.0 % | +0.0159 [0.0143, 0.0180] | 2.90 [2.64, 3.19] | 1.0072 | KLD |

**8,000 MiB** (delta 0: hit 90.4 %)

| Delta | Decode gain | Hit | Swaps | dKLD | Same-top drop (points) | PPL ratio | Rule |
|---|---|---|---|---|---|---|---|
| 0.00025 | +1.1 % | 91.1 % | 0.4 % | +0.0003 [−0.0001, 0.0006] | 0.21 [0.02, 0.38] | 1.0008 | slow |
| 0.0005 | +0.8 % | 91.9 % | 0.8 % | +0.0007 [0.0004, 0.0010] | 0.30 [0.12, 0.46] | 1.0001 | slow |
| 0.001 | +3.7 % | 92.8 % | 1.4 % | +0.0017 [0.0013, 0.0021] | 0.49 [0.30, 0.68] | 1.0011 | slow |
| 0.0025 | +6.9 % | 94.6 % | 2.9 % | +0.0041 [0.0035, 0.0047] | 1.04 [0.82, 1.23] | 1.0011 | **pass** |
| 0.005 | +8.3 % | 96.5 % | 4.5 % | +0.0074 [0.0064, 0.0085] | 1.64 [1.43, 1.85] | 1.0022 | KLD |

Draft acceptance stays at 0.72 to 0.75 in every arm. Perplexity never moves more than 0.8 %, so it decides nothing; KL divergence
and the same-top drop are the sensitive measures. At 3,000 and 4,500 MiB the larger deltas' gains rest on two runs each, so they are
less precise (about ±5 %).

What the curve says:
- **Quality cost per unit of delta falls as the cache grows.** At 8,000 MiB fewer experts sit outside the cache, so fewer picks can
  flip. That is why the largest cache passes the largest delta.
- **Each point of hit rate is worth about 1.5 to 2 % of decode speed at every cache size.**

## GSM8K

All 1,319 test questions; thinking off; greedy; at most 768 tokens; served settings with MTP on; a fresh server per arm; the same
questions for both arms.

| Cache | Delta | Delta 0 right | Pick right | Difference [95 % CI] | Only delta 0 right / only pick right | McNemar p | Cut at 768 tokens |
|---|---|---|---|---|---|---|---|
| 3,000 MiB | 0.001 | 1,250 (94.77 %) | 1,245 (94.39 %) | −0.38 points [−1.21, +0.45] | 18 / 13 | 0.47 | 42 / 47 |
| 8,000 MiB | 0.0025 | 1,254 (95.07 %) | 1,247 (94.54 %) | −0.53 points [−1.29, +0.23] | 16 / 9 | 0.23 | 41 / 42 |

Neither pick is withdrawn: neither answers fewer questions correctly than delta 0 at p < 0.05. Both point estimates are slightly
negative, about half a point. That fits the KL divergence cost, and a test of this size cannot rule it out: the intervals reach
−1.2 to −1.3 points. The pooled difference over both cache sizes is −0.45 points (34 against 22 discordant questions, McNemar
p = 0.14). 4,500 MiB had no pick, so it had no GSM8K run.

## Files

- **Rule and amendment:** `results/PREREGISTRATION.md`, `results/AMENDMENT-1.md`.
- **Analysis output:** `results/final/`:
  - `speed.txt` / `speed.json`: final sweep;
  - `speed-confirm.txt` / `.json`: the 5-block measurement at 3,000 and 4,500 MiB;
  - `quality.txt`: paired KL divergence;
  - `rule.txt` / `rule.json`: the rule applied to the final sweep;
  - `gsm-*.txt`.
- **Raw runs:** `results/runs/<stamp>-spd-*` (final sweep), `results/runs/<stamp>-spdc-*` (5-block measurement) and
  `results/runs/<stamp>-gsm-*`, each with per-prompt decode results, GPU telemetry and a manifest (engine diff and binary hashes);
  `results/kld/q16-*`, with per-position dumps.
- **Scripts:** `scripts/speed_compare.py`, `speed_blocks.py`, `paired.py`, `apply_rule.py`, `gsm8k.py`; run lists in `sweeps/`.
- **Engine changes:** `patches/0001-route-bias-mode-mul.patch` and `patches/0002-route-stats-layers-drop-kld-dump.patch`, on
  llama.cpp d02b4c0d with glm53-offload b5e86ee applied. `patches/0003-cpu-moe-work-share.patch` (CPU work sharing, on by default
  in `scripts/serve.sh`) came after these measurements: it leaves the output byte-identical and the median speed unchanged, and keeps
  speed steady from one server start to the next. `scripts/setup.sh` applies all four.
