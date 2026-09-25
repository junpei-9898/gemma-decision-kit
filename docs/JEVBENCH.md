# JevBench public-subset measurement

Measured 2026-09-25 with Gemma Decision Kit v0.7.0. This is an external public-reference evaluation, not the official sealed-set JevBench score or a new model-adoption certification.

## Scope and comparison

All rows use the same 231 public task IDs: original 72, easy 48, hard 111. External values are published results, not local reruns. Their source is the [v1.3.0 per-task artifact](https://github.com/fstandhartinger/jevbench/blob/1bcc55eb6c8cffde2306b3db03ede39b61c6152a/results/v1.2/jevbench-v1.2-per-task.json). They must not be substituted for v1.4.2 sealed-set leaderboard scores.

| System | Total /231 | Original /72 | Easy /48 | Hard /111 |
|---|---:|---:|---:|---:|
| **This kit: Eider + optimized vLLM, Gemma 4 NVFP4** | **205** | 70 | 48 | 87 |
| DeepSeek V4.1 Flash (thinking default) | 226 | 71 | 48 | 107 |
| GPT-5.6 Luna (low reasoning effort) | 225 | 70 | 48 | 107 |
| OpenJev (thinking, BF16) | 205 | 72 | 48 | 85 |
| Jev 1.13.0 | 200 | 71 | 48 | 81 |
| djev | 194 | 71 | 48 | 75 |
| SemIf Qwen3.5-4B | 187 | 71 | 48 | 68 |
| Laya (ModernBERT-large 421M) | 135 | 50 | 46 | 39 |

Kit accuracy is 88.74%, with a scenario-group bootstrap 95% interval of 84.51–92.80% (3,000 resamples, seed 89). Versus Jev, 10 tasks are correct only for the kit and 5 only for Jev. The paired difference is +2.16 percentage points, with a 95% interval of −0.88 to +5.38: a clear advantage is not established. Three passes measure repeatability, not 693 independent accuracy examples. No prompt or threshold tuning used these predictions.

## Timing and probability quality

| Pass | Requests | Correct | p50 | p95 |
|---|---:|---:|---:|---:|
| First, including first request | 231 | 205 | 84.49ms | 504.22ms |
| Replay 1, same order | 231 | 205 | 83.82ms | 497.90ms |
| Replay 2, same order | 231 | 205 | 84.38ms | 496.70ms |

One Edge Xpert GB10, Linux ARM64, `spark` profile, NVFP4 weights / FP8 KV. One question per request, strictly sequential through the unchanged public loopback HTTP server. Inputs: 123–3,958 physical tokens, median 202; 159,286 tokens per pass. Model loaded once, existing compilation caches reused, default prefix cache retained; repeated inputs can reuse prefixes. No truncation. The first request took 1.045s; the remaining 230 first-pass requests had p50 84.46ms / p95 499.71ms. Loading took 135.54s separately; total owned-container wall time was 251.73s. These are deployment measurements, not a hardware-neutral speed ranking or an Internet API speedup claim.

All 693 responses passed schema validation. Both replay passes matched all 231 first-pass answers and probabilities exactly. On the 231 distinct tasks, Brier is 0.20913 (multiclass sum; binary two-class convention), ECE is 0.09502 (10 equal-width bins), and ordinal expected-value MAE is 0.16685 over 18 score tasks. Mean total-variation distance on the 10 exact-gold-distribution tasks is 0.39525. **21 of the 26 wrong answers had confidence at least 0.9**; probabilities remain uncalibrated. External per-item probability vectors were unavailable in the outcome artifact, so no matched calibration superiority is claimed. Cost is unmeasured, not zero.

## Fixed versions and input mapping

- Kit commit: `c263c4fdb3368c68d9d860f406cbb161840896e0` (v0.7.0).
- Eider compiler/finisher: `bf42ac73bcc0718d9b86cb48cd590e332efd14af`.
- Model: `nvidia/Gemma-4-26B-A4B-NVFP4`, revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6`.
- vLLM: `0.26.1.dev0+gf2654939e.d20260726`; model context 65,536, request limit 65,535.
- Runtime image: `sha256:19627342e1da2607f4db50745dca30e57d7dd0ebff06062f03fd69b43a252931`.
- JevBench: `1bcc55eb6c8cffde2306b3db03ede39b61c6152a`.
- Canonical dataset hash: `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`.

Published labels were retained unchanged, not newly human-certified. The 35 structured states were losslessly serialized as JSON text because this distribution accepts text state. Questions, criteria and their order were retained; expected answers, provenance and rationales were excluded from inference inputs. `noul` maps to yes/no probabilities; choice/score use Eider distributions. Upstream scoring uses argmax for score accuracy and probability-weighted expectation for ordinal MAE. No new Japanese or media tests were run.

## Execution caveat

Benchmark execution and request/result integrity passed, but the original supervisor completion gate is **FAIL**. A single PID-ownership warning occurred after all 693 responses and the runner completion record were saved. All earlier monitor samples showed no competing GPU/container workload. The supervisor sent SIGTERM only to its owned container, which exited with code 0; no GPU work remained. A teardown sampling race is plausible, but the cause is **UNVERIFIED**. The run was not repeated to conceal this failure, and it is not described as an unconditional operational pass.

The benchmark record is WI-089. HTTP requests (693), backend decision-sequence calls (693), physical tokens (477,858) and resident wall time are distinct counters. Internal startup/chunk forward counts and kernel-active GPU seconds were not instrumented. The reported data does not establish other-GPU performance, calibrated confidence, or an official leaderboard position.
