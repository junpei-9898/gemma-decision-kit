# Beyond three-choice: typed-output comparison (WI-060)

**Research evaluation, not a new public API contract.** Gemma Decision Kit v0.3.0 still accepts exactly three choices. Its Gemma row below uses a research bridge that projects A–H candidate tokens with the existing optimized backend, interprets true/false probabilities and computes an ordinal expectation. No checkpoint, quantization or attention/MoE kernel was changed. These measurements do not mean `gemma-decision predict/serve` now accepts these types. SemIf also uses an explicit binary/score bridge over its public choice scorer. Eider Qwen, Laya and NanoJev use their native typed contracts; DiffusionGemma’s native contract was audited but execution was blocked; NanoJev calls binary questions `boolean`.

## Quality by output type

Correct against frozen AI-provisional labels; **not audited human-gold accuracy**.96case IDs across26correlated groups. Noul and choice2 intentionally reuse20binary propositions. Choice4 has20routing cases, choice8 has16routing cases, score has20cases over four rubrics. Simple short fictional Japanese examples, not a representative production or long-document test. No failed question, wrong answer or ambiguous output is dropped. No model-specific prompt tuning or label changes after inference.

| Configuration | Choice 2 | Choice 4 | Choice 8 | Noul / boolean | Score: top level |
|---|---:|---:|---:|---:|---:|
|Gemma4 NVFP4 · research adapter|18/20|20/20|16/16|18/20|16/20|
|Eider Qwen3.6 NVFP4|18/20|20/20|16/16|18/20|18/20|
|SemIf + Qwen3.5-4B · adapter|18/20|20/20|16/16|18/20|15/20|
|Laya multilingual|14/20|17/20|16/16|15/20|7/20|
|NanoJev root checkpoint|10/20|17/20|13/16|10/20|7/20|
|DiffusionGemma NVFP4|BLOCKED|—|—|—|—|

| Configuration | Boolean Brier ↓ | Score MAE ↓ (0–4) | Mixed correct /40 |
|---|---:|---:|---:|
|Gemma4 NVFP4 · research adapter|0.1000|0.2121|40/40|
|Eider Qwen3.6 NVFP4|0.0836|0.2140|40/40|
|SemIf + Qwen3.5-4B · adapter|0.1031|0.3465|40/40|
|Laya multilingual|0.1973|1.0537|30/40|
|NanoJev root checkpoint|0.2700|0.8929|20/40|
|DiffusionGemma NVFP4|—|—|—|

Choice accuracy is exact-label agreement. Binary decisions use fixed p(true)>=0.5; Brier is mean squared probability error. Score has five ordered levels0–4; top-level accuracy uses argmax, while MAE uses the returned probability-weighted expected index. Score is an ordinal rubric, not arbitrary numeric generation. Small differences of one or two cases do not establish general superiority. Laya's released type/option-count temperature calibration is retained; the other raw distributions are not certified calibrated for this suite.

## Warm latency (milliseconds)

| Configuration | Choice 2 | Choice 4 | Choice 8 | Noul / boolean | Score | Mixed 8 questions |
|---|---:|---:|---:|---:|---:|---:|
|Gemma4 NVFP4 · research adapter|59.4|60.4|63.5|60.2|61.0|518.1|
|Eider Qwen3.6 NVFP4|182.5|178.4|188.2|183.1|187.8|851.3|
|SemIf + Qwen3.5-4B · adapter|69.4|79.0|83.0|69.6|81.0|655.2|
|Laya multilingual|6.0|6.0|6.7|6.2|6.7|16.7|
|NanoJev root checkpoint|18.3|22.1|26.6|15.7|20.8|160.1|
|DiffusionGemma NVFP4|BLOCKED|—|—|—|—|—|

Each single-question cell is the median of15calls: three fixed short cases/type ×five repeats, after warmup and the quality pass. It is not15independent quality examples. The final column is the median of five repetitions of one eight-question mixed workflow; its /40score is five repeats of eightquestions. All incorrect answers remain included. Per-type p95 and repeated-case correctness are in the JSON. Source text length and prompt counts are in the released inputs/raw responses; these are different inputs from the historical three-choice benchmark, so do not read the new numbers as an additional speed improvement.

Five configurations completed sequentially; DiffusionGemma was blocked before inference on the same Edge Xpert / NVIDIA GB10, Linux ARM64,8CPU quota,4OMPthreads. No foreign workload overlap was observed by the supervisor. Timing excludes model loading, explicit coverage audits and disk recording; includes tokenization, inference and postprocessing. Gemma/SemIf/Laya/NanoJev use CUDA-synchronized Python calls; Eider Qwen additionally includes loopback HTTP overhead; the blocked DiffusionGemma trial was planned with the same HTTP boundary. Different templates, quantization, model sizes, type heads and native batching mean these are whole-configuration comparisons, not a controlled model-only ranking.

Gemma and SemIf direct execute the mixed questions sequentially. Laya and NanoJev retain native grouped processing, and the native servers retain their own question-sharing/chunking behavior. Do not divide a mixed-workflow time and call it isolated request latency. No simultaneous independent-client throughput test was run.

## Coverage, failures and numerical caveats

All96quality cases for each of the five completed configurations have recorded outputs. DiffusionGemma has zero inference observations; its quality and speed are UNVERIFIED, not0%. Laya's state/head/option truncation was explicitly audited with its original1024/256 limits; all tested short inputs had zero truncations. This does not remove the previously documented long-input truncation limitation. SemIf enforces its8192prompt limit, NanoJev checks full candidate paths, Gemma checks complete prompt IDs against returned IDs. Native server inputs are short and within their8192context with usage recorded; their timing includes their native wrappers.

The first Gemma research warmup failed because a tokenizer mapping was passed instead of its input_ids. It produced no quality prediction. The corrected adapter ran under a new trial; no prompts/labels changed. The first Qwen launch used an unsupported health URL; its model-list endpoint was verified and used on restart, with zero inference requests in the failed launch. A first DiffusionGemma load also stopped on one page of global swapout, with zero own-cgroup swap and no quality outputs. The GPU worker stopped; a stuck owned CPU launch parent was terminated after verifying no GPU processes remained. A fresh trial used the same fixtures/settings after full cleanup and a30second stable resource check; the same one-page global-swapout stop recurred. Both trials stopped before any inference. The second trial exited normally through the repaired cleanup handler. No third attempt or relaxation of the stop threshold was made. Free memory at the stops exceeded88GiB and own-cgroup swap was0, so these stops do not establish that DiffusionGemma exhausted memory; their cause remains unproven. These failure records and resource costs remain in the audit; no accuracy-driven retry was performed.

A float32 probability sum can slightly exceed1. The raw Gemma expected score for one maximum-level quality case was4.000000000036522; the same mixed-workflow endpoint was4.0000000065268475. A uniform1e-5 scalar endpoint tolerance is used for mathematical scoring for every model; raw values are neither clamped nor renormalized, and strict-range violations are reported separately. The research bridge therefore still needs strict output-bound handling before production API adoption. No numerical precision, calibration or exact-range guarantee is inferred from a correct argmax.

## Pinned configurations

- Gemma: NVIDIA Gemma-4-26B-A4B-NVFP4, revision a19cfe00be84568a6867111c9a68c9c44fdcffe6; kit baseline9212be4; existing optimized vLLM0.26.1.dev0+gf2654939e.d20260726, FP8KV3GiB,64Kcontext, prefix cache disabled. Only research candidate-count/type conversion differs.
- Eider Qwen: Eider bf42ac73bcc0718d9b86cb48cd590e332efd14af; NVIDIA Qwen3.6-35B-A3B-NVFP4 revision491c2f1ea524c639598bf8fa787a93fed5a6fbce; prior prepared NVFP4 artifacts;8Kcontext, retained prefix0.
- SemIf: ca3ba65f142967030ecb453346e94d6f476a69df; Qwen/Qwen3.5-4B revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, BF16, direct scorer.
- Laya: d113dca2512fb3eaca313534bc54c7162d87c1d4; convaiinnovations/laya-multilingual revision052592a15d198d9ad47da779604259b10b47b7aa; original multilingual checkpoint/config.
- NanoJev:71a513bb0163b5634467842b523ee0c0ed6fb1c7; C-Tianyu/NanoJev revision4a19595eada0857133c0d2be024f879a4077054b, root checkpoint, BF16forward/autocast, FP32parameter storage, max_length8192. This is the previously evaluated navigation-trained checkpoint, not a new Japanese specialist variant.
- DiffusionGemma: nvidia/diffusiongemma-26B-A4B-it-NVFP4 revisionec4ff3df205028f4e81c954c2227f9312b3ec2ea; existing vLLM PR407326b735c289a5f079a000a1b34c2e4be6b94c/runtime, canvas64, samples1, steps1, think0,8Kcontext, KV4GiB, prefix cache disabled. Its Jev-shaped `/v1/systemone` response converts score to zero-based indices; the wrapper's internal one-based labels are not scored directly.

No new models downloaded or versions upgraded. EXL3 is a historical alternative Gemma backend, excluded from this active NVFP4 distribution comparison. Images, video, audio and long contexts were not evaluated in this new typed suite. Final adoption remains a user decision.

## Evidence and reproduction

[Summary JSON](typed-output-results.json), [all480quality results](typed-output-cases.csv), [verification/accounting](typed-output-verification.json), [frozen cases](../benchmarks/typed/cases.json), [labels](../benchmarks/typed/gold.json), [manifest](../benchmarks/typed/manifest.json), and [raw measured calls](../benchmarks/typed/results/). The research runner/adapters are in the same benchmark folder. They expect the documented pinned runtimes and pre-provisioned `/work/imported` models/sources/dependencies, `/work/package/src` for the kit and `/gemma` for its checkpoint; native server stack additionally expects `/work/assets` for the previously prepared diffusion runtime and `/work/eider-serve`. They do not download or provision those dependencies. Run each model in an isolated, offline container, using the runner for direct models and the stack launcher for native HTTP models. These are evidence scripts, not an additional supported deployment command. Model weights and private evaluation data are not redistributed.
