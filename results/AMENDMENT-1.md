# Amendment 1 to the pass rule: re-measuring speed at 3,000 and 4,500 MiB

Written 2026-11-18 at 19:50 UTC, after the final speed and quality sweeps were read and before any confirmation run.
`PREREGISTRATION.md` is unchanged (sha256 f1ec892320bb3e6bcd0a599260c2da29f710b72425f2822d13f807b154cc63aa).

## Why

The registered speed threshold (gain >= +5 %) assumed that two runs of one setting differ by about 2 %. That held at 8,000 MiB: each
delta-0 run read against the other five came out between -1.09 % and +0.30 %. It did not hold at the smaller caches. At 3,000 MiB the
six delta-0 runs averaged 129.1, 145.2, 144.6, 144.9, 145.3 and 129.2 tok/s (-9.0 % to +4.8 % one against the rest); at 4,500 MiB
157.3, 150.6, 170.2, 172.4, 158.2 and 171.7 tok/s (-9.3 % to +6.7 %). These swings are larger than the gaps between the deltas the
rule has to choose between.

What we found:

- The slow state is set for a whole run (the warmup request is already slow) and is not tied to the arm. Hit rates are identical
  across repeats, and so are GPU clocks (about 2,770 MHz).
- Eight back-to-back starts at 3,000 MiB on an otherwise idle machine were all fast: 147.3 to 152.4 tok/s on one 512-token request.
- The slow runs come in clusters. The system journal shows other programs at those times, for example:
  - the Arch keyring sync timer during the slow delta-0 run at 17:18;
  - a snapshot cleanup at 17:03;
  - desktop programs (Discord, Brave).
- The server is sensitive to this. Its main thread shares core 0 with 38 other server threads, the 16 compute threads wait for each
  other at every step, and the CPU's share of the expert work grows as the cache shrinks.

## Outcome as registered (recorded before the re-run)

The rule applied as written (`results/final/rule.txt`):

| Cache | Pick | Gain | dKLD |
|---|---|---|---|
| 3,000 MiB | delta 0.001 | +8.94 % | 0.00401 |
| 4,500 MiB | delta 0.0005 | +7.04 % | 0.00207 |
| 8,000 MiB | delta 0.0025 | +6.91 % | 0.00409 |

## What changes

Only the speed measurement at 3,000 and 4,500 MiB. The thresholds, the quality numbers (deterministic) and the 8,000 MiB result
stay as registered.

- **Settings.** Every delta that passes the quality bounds at these two cache sizes: 0 (the reference), 0.00025, 0.0005 and 0.001.
- **Runs.** 5 blocks. Each block runs the four settings back to back in a shuffled order (seed 20261118; `sweeps/speed-confirm.txt`),
  with the two cache sizes alternating block by block. Each run uses a fresh server and the same 24 prompts as before.
- **Statistic.** Block gain = the mean over the 24 prompts of (arm decode rate / decode rate of that block's delta-0 run) - 1. A
  setting's gain is the median of its 5 block gains. The registered statistic over these runs (all of an arm's runs against all
  delta-0 runs) is reported alongside.
- **Background load.** Busy CPU cores outside the server are logged once a second (`scripts/bgcpu.py`). The log is a diagnostic only:
  no run is excluded.
- **The rule.** Applied unchanged to the median gains: gain >= +5 %, dKLD <= 0.0045, perplexity within 1 %, and a gain within 1 point
  of a smaller passing delta goes to the smaller delta. The GSM8K confirmation then follows `PREREGISTRATION.md` for the resulting
  picks.
- **Reporting.** Both outcomes, as registered and as amended, are reported.
