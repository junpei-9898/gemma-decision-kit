> Historical v0.6 documentation; commands and defaults below are not current. See [current README](../../README.md).

> v0.6.0: adds opt-in actual pinned **Eider decision semantics + optimized vLLM**. Build the CPU bridge first: [Eider setup and API migration](../../docs/EIDER.md). Select `--semantics eider` for the new integration. The default remains `legacy` because a combined AV regression check failed. Historical benchmark tables below retain their original versions and are not automatically measurements of this new path.

[Current integration validation and known AV failure](../../docs/EIDER_RELEASE_VALIDATION.md): text61/64 with exact prior-answer parity; image9/9 and short-video3/3 synthetic checks. Combined AV reasoning is experimental and has a recorded error.

The [Spark / GB10 OEM tuning and standard GPU policies](../../docs/HARDWARE.md) are selected with `--hardware auto|spark|standard`. Other NVIDIA GPU hardware remains unverified.

# Gemma Decision Kit

Local **NVFP4 Gemma 4 decisions for text, images, short videos and audio**: Eider `choice`, `noul` and `score` outputs with **uncalibrated probabilities**, without prose generation. v0.3.0 extends text input to the native256K context (262143input tokens); the default is64K. NVFP4 only; EXL3 remains in historical v0.1.0. [Context setup and measured accuracy](../../docs/CONTEXT.md).

Experimental release. Actual Eider CPU decision semantics with a project vLLM/media adapter; not an official Jev clone or a complete drop-in Eider API. No new model training, no bundled weights. [日本語](../../README.ja.md).

[Illustrated Eider / vLLM / Gemma guide (Japanese)](../../docs/EIDER_VLLM_GUIDE.ja.md) explains the engines, this kit’s implementation and audiovisual processing with seven diagrams. [Offline HTML edition](../../docs/EIDER_VLLM_GUIDE.ja.html).

## Automatic input processing (v0.5.0)

Use `analyze --source recording.mp4` with your existing question JSON. Text/image/audio/video
routing is automatic. Video with speech pairs MOSS utterances and visual windows on the source
timeline, with at most10seconds per window; no manual `--media`/`--audio` selection. Results
include coverage, local provisional judgments and explicit failures. Multi-window final decisions
aggregate local choices; this is not unrestricted joint reasoning over every original frame.

```sh
gemma-decision analyze --model-path /models/nvfp4 \
  --source /input/recording.mp4 --input examples/request.json \
  --audio-model-path /models/moss --audio-python /state/moss-env/bin/python \
  --output /state/analysis.json
```

`serve-input` adds a serialized loopback `/v1/analyze` endpoint for bounded inline files; larger
recordings use CLI. Existing `predict`/`serve` remain compatible. [Setup and limits](../../docs/UNIFIED_INPUT.md)
· [Validation status](../../docs/UNIFIED_VALIDATION.md).

## Supported inputs

| Input | Processing | Interface |
|---|---|---|
| Text | Eider choice / noul / score | `predict` / HTTP |
| Image | Visual analysis of one PNG/JPEG | `--media` with image JSON / HTTP |
| Video | Sampled frames from one MP4, up to10seconds | `--media` with video JSON / HTTP |
| Audio / video audio track | MOSS transcription, anonymous speakers and utterance times, optionally passed to Gemma; up to30minutes | `transcribe` / `predict --audio` (CLI/Python only) |

v0.4.0 bundles the audio adapter. MOSS weights, audio dependencies and FFmpeg require separate
setup. MOSS converts speech to text; Gemma does not gain a native audio encoder. Legacy `predict --media` does not automatically process audio; `analyze --source` does. **v0.5.0 has actual short audiovisual pipeline checks. Generic model aggregation made an
incorrect final choice; explicit `any`/`all` semantics provide logical aggregation.**
[Audio setup](../../docs/AUDIO.md) · [Validation status](../../docs/UNIFIED_VALIDATION.md).

## Accuracy and speed on our Japanese decision task

**93.3% label agreement (112/120) and ~80ms warm short-question latency on GB10.** This was the highest agreement among the six tested configurations below. In the same-node HTTP comparison, short-question latency was **1.62× faster than DiffusionGemma**, with **+5.0percentage points** higher v1 agreement.

### Measured hardware

| Item | Test environment |
|---|---|
| Machine / GPU | Edge Xpert, one NVIDIA GB10 (Blackwell, SM121) per run |
| Memory | CPU/GPU unified memory; OS-visible approximately121.6GiB, **not dedicated model VRAM** |
| Platform | Linux ARM64; kit uses PyTorch2.11.0+cu130; other runtimes pinned per configuration |
| Study-A CPU allocation | Container quota8CPUs, `OMP_NUM_THREADS=4` |

A more powerful GPU **may reduce latency**, especially for compute-heavy long-input prefill, but we have not measured a speedup factor on other GPUs. CPU preprocessing, memory bandwidth and compatible kernels also matter; do not multiply these timings by advertised TOPS or bandwidth ratios. The pinned ARM64/GB10 image is not a validated x86/RTX deployment.

### Model memory footprint

| NVFP4 mode | Peak live GPU tensors | Peak allocator-reserved memory |
|---|---:|---:|
| Optimized text |21.99GiB|23.91GiB|
| Images/video (`--media`) |23.11GiB|24.10GiB|

Measured in the v0.2.0 GB10 regression; media peaks include initialization profiling. Reserved memory **includes** live allocations: do not add the columns. These are PyTorch GPU measurements, excluding some driver/host allocations, not minimum VRAM certifications. GB10 shares system memory between CPU and GPU, so allow additional room for the OS, runtime and loading. The machine's121.6GiB capacity is not the model requirement; conversely, these results do not prove that a24GB discrete GPU can run this package. [Memory definitions and evidence](../../docs/MEMORY_AND_LONG_INPUT.md).

These are fixed Japanese evidence judgments (`supported / refuted / insufficient`) against **AI-provisional labels**, not human-certified accuracy or a general leaderboard. Quality uses120case IDs; latency uses one54-character state repeated20times. Fast answers to that one example do not imply high corpus accuracy.

| Configuration | Agreement (v1,120cases) | Short-question median | Study |
|---|---:|---:|---|
| Gemma Decision Kit · Gemma 4 26B-A4B NVFP4 | **93.3% (112/120)** | **80.4ms** | A |
| DiffusionGemma 26B-A4B NVFP4 | 88.3% (106/120) | 130.6ms | A |
| Eider · Qwen3.6-35B-A3B NVFP4 | 85.8% (103/120) | 250.3ms | B |
| SemIf · Qwen3.5-4B BF16 | 64.2% (77/120) | 119.6ms | B |
| Laya multilingual · 0.322B † | 39.2% (47/120) | 8.8ms | B |
| NanoJev · 0.6B navigation checkpoint | 38.3% (46/120) | 40.6ms | B |

**A:** same GB10 node,2026-09-23, loopback HTTP. **B:** earlier2026-09-20 runs on a different GB10 node; Qwen uses HTTP, others native Python APIs. All are loaded/warm medians, excluding startup. Model sizes, quantization, templates and API boundaries differ: this is a configuration comparison, not a controlled model-only speed ranking. Current-kit rows measure its frozen optimized recipe through the evaluation HTTP adapter; v0.2.0 verifies prediction parity, not an identical public-server latency guarantee.

**† Laya input coverage:** its8.8ms result is for the54-character short probe: **no truncation**,20/20matching labels on repetitions of one question. The2099/8108-character probes instead truncate5/5requests each, take21.6/23.1ms, and match0/5labels each; these are **not full-document latencies**. All100v2quality prompts also truncate the target-claim-containing head. The120v1cases used in the table do not truncate, so their39.2%agreement must not be explained as a truncation artifact. [Coverage counts](../../docs/MEMORY_AND_LONG_INPUT.md).

Laya and NanoJev are faster on this short input but have lower label agreement on this task. DiffusionGemma wins the8-question workflow:377.6ms vs732.8ms, with40/40 vs35/40 matching labels across5repeats of the same8questions. The tested NanoJev checkpoint targets navigation, not general Japanese evidence classification. No blanket claim of being faster or more accurate than every OSS model is made.

[Full comparison, instruction sensitivity and immutable versions](../../docs/COMPARISON.md) · [Machine-readable aggregates](../../docs/comparison-results.json) · [Text/image/video timings and startup limits](../../docs/BENCHMARKS.md)

### Long-input optimization history

**Historical native-runner measurements.** These fixtures used65536internal context. v0.3.0 separately validates the public API up to262143input tokens; see [new HTTP measurements and accuracy by length](../../docs/CONTEXT.md). Do not treat the historical timings below as identical public-server timings.

| State characters / prompt tokens | Stabilized baseline | Final recipe | Time reduction |
|---|---:|---:|---:|
|10,000 / 7,138|1.563s|1.319s|15.6%|
|30,000 / 21,197|6.569s|4.570s|30.4%|
|50,000 / 35,257|15.248s|8.686s|43.0%|

Same GB10, pinned Gemma4 NVFP4 weights and full input fixtures, one question, loaded/shape-warmed, no prefix-cache reuse,3repeats per length. Includes input preparation + native inference; excludes model load, HTTP and profiling. Each length is one document, not a representative long-document benchmark. Improvement includes attention launch tuning, MoE buffer/copy handling and compact ABC projection.

The baseline is the **stabilized native CUTLASS** recipe. The earlier FlashInfer path was13.714s at50kcharacters (vs final8.686s,36.7% shorter), but had different stability/quality; at10k it was1.242s and faster than final1.319s. We do not claim every input got faster versus every historical recipe. [Stages, quality checks and limits](../../docs/MEMORY_AND_LONG_INPUT.md).

## Installation

Clone this repository, then follow [the pinned GPU installation guide](../../docs/INSTALL.md). Validated on NVIDIA GB10 / Edge Xpert, Linux ARM64, CUDA13. Other GPU architectures are unverified. CPU contract tests alone do not certify GPU inference.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
gemma-decision info
gemma-decision validate --input examples/request.json
python -m unittest discover -s tests -v
```

## Use

After installing the pinned runtime and downloading the NVFP4 checkpoint:

```sh
# Original optimized text path (no vision encoder).
gemma-decision predict --model-path /models/nvfp4 --input examples/request.json
# Text + image/video service, one NVFP4 model with its vision encoder.
gemma-decision serve --media --model-path /models/nvfp4 --port 8765
curl http://127.0.0.1:8765/v1/decisions \
  -H 'Content-Type: application/json' --data-binary @examples/image-request.json
```

`--profile speed` is retained for compatibility and is the only profile. `--media` selects the separately validated vision recipe at startup. It does not simultaneously load a second model. Restart to switch modes. Text-only mode retains the original performance recipe; its speed/precision figures do not automatically apply to text served in media mode.

Input: text `state`, 1–64 named questions, optional inline media. Opt-in Eider semantics support choice (2–64 options), noul (true probability) and score (expected rubric position); [exact schema and migration](../../docs/EIDER.md). `--semantics legacy` retains the old three-choice API. All questions run sequentially. Text defaults to65535input tokens per question, with opt-in131071/262143limits; media mode remains8192. Expanded media and every question are checked before inference; no truncation. Probabilities are not calibrated correctness guarantees.

HTTP binds only127.0.0.1. Use an authenticated tunnel for remote use. This is a small local serialized API, not an Internet production gateway. Inline audio/video is available through `/v1/analyze` with per-request model loading. `/v1/decisions` is resident text/visual inference. No Gemma free-text generation, dynamic GPU batching, tensor parallelism or LoRA. `--media` video uses sampled-frame visual analysis; use the separate `--audio` path for speech.

## Evidence and license

[Benchmarks and validation](../../docs/BENCHMARKS.md) separate native historical speed from packaged API timings. Small synthetic media fixtures do not certify real-world OCR or arbitrary video accuracy.

Code and project-authored examples: Apache-2.0. See [NOTICE](../../NOTICE), [license audit](../../docs/LICENSES.md), [LICENSE](../../LICENSE). Model weights and runtime/container dependencies are obtained separately under their own licenses.

## Context length and accuracy (v0.3.0)

![Measured accuracy and latency across context lengths](../../docs/assets/context-accuracy.png)

Larger supported input is **not a promise of constant accuracy**. The blue series measures the same9synthetic cases at each length with varied background records; the orange series retains the earlier7repeated-negative stress cases. These are frozen AI-provisional labels, separate from the93.3% (112/120) historical comparison. All errors remain in the results. Long-context evidence retrieval, instruction robustness and label auditing are **future accuracy work**; no accuracy fix is claimed in this release. See [counts, methods, limitations and raw data](../../docs/CONTEXT.md).

Observed on the9paired cases: short**7/9 (77.8%)**, near256K**7/9 (77.8%)**. This is a small provisional-label study; one label has interpretive ambiguity pending independent review. Do not infer a universal or monotonic accuracy curve.64K/128K/256K allocate3/4/8GiB KV; see [total measured memory](../../docs/CONTEXT.md#runtime-and-memory).


## Local audio (v0.4.0)

MOSS provides text, anonymous speakers and utterance timestamps. Use `transcribe` alone or `predict --audio` to feed Gemma4. MOSS exits before Gemma loads. HTTP audio is not supported. Adapters, a pinned manifest and notices are bundled; weights and meeting material are not.

After [audio setup](../../docs/AUDIO.md), pass a local recording and your existing question JSON:

```sh
gemma-decision predict --model-path /models/nvfp4 --input examples/request.json \
  --audio /input/recording.mp4 --audio-model-path /models/moss \
  --audio-python /state/moss-env/bin/python --transcript-output /state/transcript.json
```

[Commands, limits and privacy](../../docs/AUDIO.md) · [GPU acceptance status](../../docs/AUDIO-VALIDATION.md).

## Beyond three choices: additional typed-output evaluation

**Historical measurements: the Gemma row used the earlier research adapter, not the new v0.6.0 Eider integration.** The current API exposes typed outputs, but this table is retained without relabeling its measurements. Eider Qwen/Laya/NanoJev use native typed APIs; SemIf uses a research conversion. DiffusionGemma was blocked during startup and has no new measurements. Counts below are agreement with provisional labels on96short synthetic cases (26correlated groups); score uses the most probable of five levels.

| Configuration | Choice 2 | Choice 4 | Choice 8 | Noul / boolean | Score: top level |
|---|---:|---:|---:|---:|---:|
|Gemma4 NVFP4 · research adapter|18/20|20/20|16/16|18/20|16/20|
|Eider Qwen3.6 NVFP4|18/20|20/20|16/16|18/20|18/20|
|SemIf + Qwen3.5-4B · adapter|18/20|20/20|16/16|18/20|15/20|
|Laya multilingual|14/20|17/20|16/16|15/20|7/20|
|NanoJev root checkpoint|10/20|17/20|13/16|10/20|7/20|
|DiffusionGemma NVFP4|BLOCKED|—|—|—|—|

Warm median milliseconds: three fixed cases/type ×five repeats; Mixed is the whole eight-question workflow ×five repeats. DiffusionGemma stopped on new global swapout during both startup attempts, before any inference. Same GB10 node, sequential models; Qwen/Diffusion include HTTP while others use Python APIs.

| Configuration | Choice 2 | Choice 4 | Choice 8 | Noul / boolean | Score | Mixed 8 questions |
|---|---:|---:|---:|---:|---:|---:|
|Gemma4 NVFP4 · research adapter|59.4|60.4|63.5|60.2|61.0|518.1|
|Eider Qwen3.6 NVFP4|182.5|178.4|188.2|183.1|187.8|851.3|
|SemIf + Qwen3.5-4B · adapter|69.4|79.0|83.0|69.6|81.0|655.2|
|Laya multilingual|6.0|6.0|6.7|6.2|6.7|16.7|
|NanoJev root checkpoint|18.3|22.1|26.6|15.7|20.8|160.1|
|DiffusionGemma NVFP4|BLOCKED|—|—|—|—|—|

**Accuracy and speed tradeoffs depend on output type; Gemma is not the highest-accuracy configuration in every format.** These short Laya inputs had no truncation. See [Brier, ordinal MAE, mixed-workflow correctness, numerical caveats and all cases](../../docs/TYPED_OUTPUTS.md).
