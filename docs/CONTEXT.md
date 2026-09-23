# Native context extension trial — not released

**The distributed API remains capped at8192input tokens per question.** An unreleased candidate accepted near64K complete prompts, but failed its long-input quality gate. No v0.3.0 release was published.128K and256K were **SKIPPED**, not passed or disproved.

Gemma4 26B-A4B supports262144total context tokens in the [pinned NVFP4 configuration](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4/blob/a19cfe00be84568a6867111c9a68c9c44fdcffe6/config.json). This is a model architectural limit, distinct from the package's validated input contract.1Mtokens is unsupported by this recipe. One output token must also fit inside the native context.

## Observations (2026-09-24)

| Input setting | Actual prompt tokens | Synthetic evidence cases | Same-case warm HTTP median (n=3) |
|---|---:|---:|---:|
|8k-diagnostic|8,171–8,174|4/7|1.528s|
|32k-diagnostic|32,754–32,757|4/7|7.914s|
|64k-r1|65,521–65,524|4/7|20.843s|

The candidate matched all231frozen legacy decisions **and probabilities exactly**. Its seven short controls were correct at every stage. However, repeated-filler long inputs scored4/7 at8K,32K and64K.64K supported-evidence probes at the beginning, middle and end all returned `insufficient`; refuted and absent-evidence probes passed.8K had a different error pattern: equal aggregate accuracy does not imply identical errors.8K/32K diagnostics used the candidate restricted to those input limits and the unchanged65536internal context/3GiB KV. They were not a separate old-release rerun. These findings do not establish that increasing the wrapper limit caused a regression; they do prevent claiming that long-document accuracy was validated.

The fixture is deliberately narrow: repetitive Japanese warehouse records surrounding one approval fact, or no approval fact. The repeated filler says that its individual record contains no trial-approval information, which may bias judgments toward `insufficient`; this explanation is a hypothesis, not an established cause. Labels were fixed before each stage's predictions. These seven synthetic AI-provisional cases are **not a general accuracy benchmark** or a replacement for the earlier120case comparison. The failed cases were retained, not removed or relabeled.

## Measurement scope

NVIDIA GB10 / EdgeXpert, Linux ARM64,8CPU quota/4OMP threads; pinned NVFP4 checkpoint and vLLM0.26.1.dev0+gf2654939e.d20260726, unchanged optimized kernels, FP8 KV3GiB,8192prefill chunk, one question at a time, prefix caching off. Timing is real same-container loopback HTTP including tokenization/JSON, model already loaded. The table repeats the **same supported-at-beginning fixture** three times; its answer was incorrect at all three lengths. These are processing times, not time-to-correct-answer claims. Other six long cases ran once each, with no failures excluded.32K reused the8K diagnostic process;64K ran in a separate process after legacy regression.

Near64K input contained105520–105527Japanese characters and65521–65524prompt tokens. Every accepted prompt was checked against the backend's returned token IDs; no input truncation. Over-limit HTTP requests were rejected with400. The configured65535boundary was covered by CPU contract tests; the GPU fixtures were slightly below it.64K peak PyTorch allocated memory22.005GiB, reserved23.908GiB (includes allocated); these are not minimum discrete-VRAM certifications or total host consumption. Minimum system available memory87.92GiB; no new swapout, OOM or foreign workload contention was observed.

## Disposition

The candidate's CPU contract suite passed25tests; the initial trial harness had a tokenizer return-type mistake and stopped **before inference**, then was corrected in a new recorded trial. The public package code and default limit remain unchanged. Image/video limits also remain unchanged. Candidate128K/4GiB and256K/8GiB configurations have no GPU validation and are not distributed.

Next: use a broader frozen long-document set, including varied neutral filler and natural documents, compare matched short/long judgments, and independently review labels. Keep this failed stress set visible. Only then reconsider the quality gate and larger tiers; do not fix this by truncating documents or discarding failed cases.

[Machine-readable trial summary](context-trial-results.json). No model weights or private evaluation inputs are redistributed.
