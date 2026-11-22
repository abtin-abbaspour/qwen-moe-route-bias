# Calibration, 2026-11-18

Qwen3.6-35B-A3B UD-IQ4_XS on one RTX 5070 Ti 16 GB (PCIe Gen4 x16) + Ryzen 9 7950X3D, 32 GB DDR5-5200. Engine: llama.cpp d02b4c0d +
glm53-offload patch (b5e86ee) + `LLAMA_MOE_ROUTE_BIAS_MODE=mul` (`patches/0001-route-bias-mode-mul.patch`) + swap counting
(`patches/0002-route-stats-layers-drop-kld-dump.patch`). Speed: Killy's 24 prompts x 512 greedy tokens, thinking on, MTP 2 drafts, fresh server per arm, median of the server's
per-prompt decode rate. Quality: `llama-perplexity` KL-divergence mode on wikitext-2 test, 4 x 4,096-token chunks, 8,188 scored positions.

## 1. Swap counting (`LLAMA_MOE_ROUTE_STATS`)

- With the bias on, each small-batch lookup also reads the unbiased router scores (a separate softmax node added before the router
  chain) and the host compares the unbiased top-8 with the selection: swaps, unbiased mass kept, favoured share of both sets, and how
  far below the unbiased cut each swapped-in expert sat. `=<path>` also writes per-token records (`scripts/route_records.py`).
- Checks: 0 swaps at delta 1e-30; at mul 0.5 every swapped-in expert sat < 0.5 below the cut in log score (max 0.4999); replaying
  the records reproduces the engine's selection on 51,683 of 51,683 token-layers; text and counters are identical with it on or off.
- Cost: 2-5 % decode speed (more when writing records). Timing runs keep it off; counts come from a second, identical pass.

## 2. The fused router kernel (confound found)

- The CUDA backend fuses softmax -> top-k -> weights for Qwen's router. An ADD after the softmax is part of the pattern, so the
  **additive** bias keeps the fusion: add 1e-30 is bit-identical to delta 0. The **logit-space** (mul) bias breaks it: mul 1e-30 has
  0 swaps yet different text and counters (other arithmetic), and costs ~5 % decode at 2,500 MiB.
- Consequence: each mode is compared with its own near-zero baseline for quality; speed is quoted against the plain fused delta 0.
  (On GLM the bias folds into exp_probs_b, which the fused kernel reads, so Killy's runs did not have this.)

## 3. Cache size, delta = 0 (cpu share 0.9)

| Cache MiB | Decode tok/s (median) | Hit rate (decode) | Draft acceptance | Note |
|---|---|---|---|---|
| 1,000 | 79.9 | 42.6 % | 0.368 | **corrupt output** (smallest expert group: 8 slots < 24 experts of a 3-token verify step) |
| 2,000 | 127.4 | 50.4 % | 0.725 | |
| 3,000 | 145.9 | 62.7 % | 0.741 | regime "low" |
| 4,500 | 174.9 / 173.5 | 75.1 % | 0.750 | regime "Killy-like" (his one card: 79 %) |
| 6,000 | 198.0 | 83.7 % | 0.735 | |
| 8,000 | 214.8 / 218.9 / 219.2 | 90.4 % | 0.742 | regime "natural" at 64K context |
| 10,000 | 233.8 | 95.1 % | 0.735 | 15.5 of 16.3 GB VRAM in use: the ceiling |

- Routing is deterministic across fresh servers (identical texts and hit counts); timing is not: the same arm spans 214.8-219.2 tok/s
  (2 %). Gains near Killy's 5 % threshold need interleaved repeats.

## 4. CPU share (the share of misses computed on the CPU once the cache is full)

| Cache | 0.5 | 0.75 | 0.9 | 1.0 (frozen) |
|---|---|---|---|---|
| 4,500 | 152.0 tok/s, 77.1 % | 172.8, 76.8 % | 173.5, 75.1 % | 117.1, 42.9 % |
| 8,000 | | 222.3, 92.7 % | 219.2, 90.4 % | 160.9, 70.8 % |

0.75 and 0.9 tie within noise; 0.9 (shipped) stays.

## 5. Delta -> swap rate (static replay of delta ~ 0 records; `replay-delta-map.txt`)

| Swap rate (of picks) | add delta, 3,000 | 4,500 | 8,000 | mul delta, 3,000 | 4,500 | 8,000 |
|---|---|---|---|---|---|---|
| 0.5 % | 0.000128 | 0.000128 | 0.000193 | 0.0117 | 0.0119 | 0.0182 |
| 1 % | 0.000255 | 0.000259 | 0.000403 | 0.0234 | 0.0240 | 0.0378 |
| 2 % | 0.000514 | 0.000531 | 0.000891 | 0.0472 | 0.0491 | 0.0816 |
| 4 % | 0.00105 | 0.00113 | 0.00243 | 0.0963 | 0.103 | 0.205 |
| 8 % | 0.00230 | 0.00271 | 0.0420 | 0.205 | 0.234 | 1.59 |

- At small delta the map hardly depends on the cache size: it is a property of the router's near-ties.
- The swap rate saturates at the uncached share of the unbiased picks: 8.25 % at 8,000 MiB, 21.4 % at 4,500, 32.4 % at 3,000.
- At equal swap rate the additive rule keeps slightly more unbiased mass than the logit-space rule (e.g. 4,500 MiB, ~3.6 % swaps:
  99.924 % vs ~99.917 %): it swaps where the absolute gap is small, i.e. low-weight experts.
- The replay predicted 11.5 % swaps for add 0.005 at 4,500 MiB; the perplexity run measured 11.1 %.

## 6. Quality harness (vs f16 KV, 512-token batches = every expert on the GPU, PPL 5.1575)

| Row | Same top token | Mean KLD | PPL ratio | Hit rate |
|---|---|---|---|---|
| q8_0 KV, 512-token batches | 97.215 ± 0.182 % | 0.00454 | 1.0017 | |
| 4,500, 1-token steps, delta 0 | 97.154 ± 0.184 % | 0.00498 | 1.0010 | 84.9 % |
| same, repeated | 97.154 % (bit-identical) | 0.00498 | 1.0010 | 84.9 % |
| 4,500, 3-token steps, delta 0 | 97.069 ± 0.186 % | 0.00492 | 1.0005 | 81.2 % |
| 3,000, 1-token steps, delta 0 | 97.191 ± 0.183 % | 0.00527 | 1.0000 | 75.3 % |
| 2,000, 1-token steps, delta 0 | 96.910 ± 0.191 % | 0.00539 | 1.0011 | 65.5 % |
| 4,500, 1-token steps, add 0.005 | 94.455 ± 0.253 % | 0.02121 | 1.0028 | 94.1 % (11.1 % swaps) |

- The bias engages in `llama-perplexity` with steps of up to 32 tokens (the cache serves those); 4 chunks at 1-token steps take
  ~140 s, so 16 chunks (32,752 positions) are ~10 min a row.
- The ~97.2 % floor is almost all the q8_0 KV cache; the expert split and step size add little. Bias-0 rows repeat bit for bit.
- add 0.005 (Killy's smallest value, but on probabilities of a 256-expert softmax) swaps 11 % of picks and costs 2.7 points.
