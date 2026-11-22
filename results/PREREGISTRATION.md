# Pass rule for the route bias on Qwen3.6-35B-A3B (fixed before the final results)

Fixed 2026-11-18 at 17:10 UTC, while the final speed sweep had finished 6 of its 48 arms and the final quality sweep had not
started; none of their numbers had been read. The thresholds come from this model on this machine, not from glm53-offload's GLM-5.3
results.

## What is decided

For each GPU expert cache size (3,000, 4,500 and 8,000 MiB), which additive route bias delta to run, from 0 (no bias), 0.00025,
0.0005, 0.001, 0.0025 and 0.005. The bias is additive on the router's softmax probabilities, applied uniformly to all 40 MoE layers,
to the cached experts (and those being filled), in decode steps of up to 32 tokens; the mixing weights stay unbiased. Everything else
is `scripts/serve.sh` defaults: cpu share 0.9, 64K context, q8_0 KV, MTP with 2 drafts, the glm53-offload engine switches.

## Measurements

- **Speed** (`sweeps/speed-final.txt`, `scripts/speed_compare.py`): 24 prompts (glm53-offload `bench/prompts-24.txt`), 512 greedy
  tokens each, thinking on, a fresh server per run. Each arm runs twice, in two passes (the second in reverse order), with three
  delta-0 runs per pass between them. Gain = the mean over the 24 prompts of (arm decode rate / mean delta-0 decode rate of that
  prompt) - 1, with the server's own decode rate (`timings.predicted_per_second`).
- **Quality** (`sweeps/kld-final.txt`, `scripts/paired.py`): `llama-perplexity` KL-divergence mode on wikitext-2 test, 16 chunks of
  4,096 tokens, the second half of each scored (32,752 positions), 3-token steps, against an f16-KV reference that computes every
  expert on the GPU (512-token batches). Delta 0 is scored the same way; each arm's cost is the paired difference to delta 0 on the
  same positions: added mean KL divergence (dKLD) and the perplexity ratio.

## The rule

A delta passes at a cache size when all of these hold:

1. **Speed:** gain >= +5 %. (Two runs of the same setting differ by about 2 %.)
2. **Quality:** dKLD <= 0.0045, the point estimate (its 95 % block-bootstrap interval is reported alongside). 0.0045 is the KL
   divergence that the q8_0 KV cache already costs this model against an f16 KV cache (0.00454, measured 2026-11-18), a setting the
   server runs with.
3. **Perplexity:** within 1 % of delta 0.

The chosen delta is the largest gain among the deltas that pass (a gain within 1 point of a smaller passing delta's gain goes to the
smaller delta). If none passes, delta 0 stays.

## Confirmation on a task

For each cache size with a chosen delta > 0: GSM8K test (1,319 questions, the original OpenAI release), thinking off, greedy, at most
768 tokens, served settings (MTP on), a fresh server per arm, delta 0 against the chosen delta on the same questions. The final answer
is the number after `####`, else the last number in the reply. The chosen delta is **withdrawn** (the next smaller passing delta is
tested instead, down to delta 0) if it answers fewer questions correctly than delta 0 with McNemar's exact test at p < 0.05. The
accuracy difference and its interval are reported either way.

## Reported regardless of the rule

The full curve per cache size: gain, hit rate, draft acceptance, swap rate (from the quality runs), dKLD, same-top-token difference
(paired, McNemar), perplexity ratio.
