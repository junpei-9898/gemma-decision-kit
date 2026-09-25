# Gemma Decision Kit

*Eider-powered decisions with optimized vLLM.*

**Eider typed decisions + optimized vLLM + NVFP4 Gemma 4**, for text, images, video and transcribed speech. Returns `choice`, `noul` or `score` with uncalibrated probabilities; no prose generation. [日本語](README.ja.md).

v0.7 has one decision engine: actual pinned Eider prepares questions and constructs answers; vLLM runs the model. MOSS handles speech, and the kit aligns it with sampled video windows. No bundled weights, new training, or claim of official Jev compatibility.

## Start here

1. [Install the pinned runtime and model](docs/INSTALL.md).
2. [Build the mandatory Eider CPU bridge](docs/EIDER.md) and set `GEMMA_EIDER_LIBRARY`.
3. For speech, also [install MOSS and FFmpeg](docs/AUDIO.md).

```sh
# Typed text request; model and bridge already installed.
gemma-decision predict --model-path /models/nvfp4 --input examples/eider-request.json
# A file: automatic text/image/audio/video routing.
gemma-decision analyze --model-path /models/nvfp4 \
  --source /input/recording.mp4 --input examples/request.json \
  --audio-model-path /models/moss --audio-python /state/moss-env/bin/python
```

| Interface | Purpose |
|---|---|
| `predict` / `serve` | Typed JSON; `serve` keeps Gemma resident. Add `--media` at startup for inline images/short videos. |
| `analyze` / `serve-input` | Automatic file processing, including speech; MOSS exits before Gemma loads. HTTP workers reload per request. |
| `transcribe` / `download-audio` | Optional speech preprocessing and explicit model setup. |

Hardware defaults to `auto`: `spark` optimization for GB10-based DGX Spark/OEM machines (measured on Edge Xpert), `standard` otherwise within the supported capability check. Other physical GPUs are unverified. [Hardware boundaries](docs/HARDWARE.md).

No `legacy`, `--semantics`, `--profile`, `gb10` alias or `predict --audio` path remains. Upgrading is a breaking API migration: use [v0.7 migration notes](docs/MIGRATION.md), particularly for Python clients and response fields.

## Validation and limits

The earlier GPU Eider evaluation (v0.6) retained all 64 prior answers/probabilities, with 61/64 provisional-label agreement; Spark mode median76.36ms for63warm short single-question requests (121–188tokens). Image9/9, short video3/3 and speech1/1 were small synthetic checks. **Combined audiovisual questions were1/2 versus legacy2/2 on an ambiguous fixture; this remains unresolved.** Making Eider the sole engine is a product simplification, not proof that the error is fixed. The v0.7 release was CPU/package validated; the subsequent text-only JevBench GPU evaluation is reported below. [Validation details](docs/EIDER_RELEASE_VALIDATION.md).

Audio uses transcription, not a native Gemma audio encoder. Videos are sampled; long-video output aggregates local judgments and cannot guarantee arbitrary cross-window reasoning. Multi-window score/noul is unsupported. Audio/file workflows include cold loading and do not have the short-text latency above. [Input behavior](docs/UNIFIED_INPUT.md).

Text context defaults to64K, max256K total; media input limit8192expanded tokens. No silent truncation. Long context can reduce accuracy; [measured curve and future work](docs/CONTEXT.md). Eider media/audio runs reached about23.5GiB allocated /25.4GiB reserved; these are not a minimum VRAM guarantee or proof of24GB fit.

## JevBench public subset (v0.7.0)

**205/231 correct (88.7%), with a median of 84.5ms per decision** on the [JevBench](https://github.com/fstandhartinger/jevbench) public subset, measured 2026-09-25. All three passes returned identical answers and probabilities. This is not an official JevBench score or ranking; sealed tasks are not included.

| System | Correct on the same 231 public tasks | Accuracy |
|---|---:|---:|
| **Gemma Decision Kit · Eider + optimized vLLM, Gemma 4 NVFP4** | **205/231** | **88.7%** |
| Jev 1.13.0 · published result | 200/231 | 86.6% |
| djev · published result | 194/231 | 84.0% |
| SemIf Qwen3.5-4B · published result | 187/231 | 81.0% |

External rows come from the pinned v1.3.0 public per-task artifact, not new runs on our hardware. The five-answer difference versus Jev does **not establish a statistically clear advantage**. Generative comparators scored higher; see the [fuller comparison and method](docs/JEVBENCH.md).

| Pass · 231 decisions each | Median (p50) | p95 |
|---|---:|---:|
| First pass | 84.49ms | 504.22ms |
| Replay 1 | 83.82ms | 497.90ms |
| Replay 2 | 84.38ms | 496.70ms |

One GB10 (Edge Xpert), v0.7.0, NVFP4 weights / FP8 KV, one question per request, sequential loopback HTTP; 123–3,958 input tokens (median 202). Model loaded once; existing compilation caches and default prefix cache retained. Model loading took 135.54s separately; the first decision took 1.045s and is included in the first-pass table. These local timings are not a controlled speed comparison with remote APIs.

Probabilities remain uncalibrated (ECE 0.0950; Brier 0.2091). All 693 responses passed schema validation, but a post-results shutdown-monitor warning left the supervisor gate **FAIL**; the container exited with code 0 and no GPU work remained. [Conditions, uncertainty and shutdown caveat](docs/JEVBENCH.md).

## Historical comparisons (not a v0.7 remeasurement)

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

Measured in the v0.2.0 GB10 regression; media peaks include initialization profiling. Reserved memory **includes** live allocations: do not add the columns. These are PyTorch GPU measurements, excluding some driver/host allocations, not minimum VRAM certifications. GB10 shares system memory between CPU and GPU, so allow additional room for the OS, runtime and loading. The machine's121.6GiB capacity is not the model requirement; conversely, these results do not prove that a24GB discrete GPU can run this package. [Memory definitions and evidence](docs/MEMORY_AND_LONG_INPUT.md).

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

**† Laya input coverage:** its8.8ms result is for the54-character short probe: **no truncation**,20/20matching labels on repetitions of one question. The2099/8108-character probes instead truncate5/5requests each, take21.6/23.1ms, and match0/5labels each; these are **not full-document latencies**. All100v2quality prompts also truncate the target-claim-containing head. The120v1cases used in the table do not truncate, so their39.2%agreement must not be explained as a truncation artifact. [Coverage counts](docs/MEMORY_AND_LONG_INPUT.md).

Laya and NanoJev are faster on this short input but have lower label agreement on this task. DiffusionGemma wins the8-question workflow:377.6ms vs732.8ms, with40/40 vs35/40 matching labels across5repeats of the same8questions. The tested NanoJev checkpoint targets navigation, not general Japanese evidence classification. No blanket claim of being faster or more accurate than every OSS model is made.

[Full comparison, instruction sensitivity and immutable versions](docs/COMPARISON.md) · [Machine-readable aggregates](docs/comparison-results.json) · [Text/image/video timings and startup limits](docs/BENCHMARKS.md)

### Long-input optimization history

**Historical native-runner measurements.** These fixtures used65536internal context. v0.3.0 separately validates the public API up to262143input tokens; see [new HTTP measurements and accuracy by length](docs/CONTEXT.md). Do not treat the historical timings below as identical public-server timings.

| State characters / prompt tokens | Stabilized baseline | Final recipe | Time reduction |
|---|---:|---:|---:|
|10,000 / 7,138|1.563s|1.319s|15.6%|
|30,000 / 21,197|6.569s|4.570s|30.4%|
|50,000 / 35,257|15.248s|8.686s|43.0%|

Same GB10, pinned Gemma4 NVFP4 weights and full input fixtures, one question, loaded/shape-warmed, no prefix-cache reuse,3repeats per length. Includes input preparation + native inference; excludes model load, HTTP and profiling. Each length is one document, not a representative long-document benchmark. Improvement includes attention launch tuning, MoE buffer/copy handling and compact ABC projection.

The baseline is the **stabilized native CUTLASS** recipe. The earlier FlashInfer path was13.714s at50kcharacters (vs final8.686s,36.7% shorter), but had different stability/quality; at10k it was1.242s and faster than final1.319s. We do not claim every input got faster versus every historical recipe. [Stages, quality checks and limits](docs/MEMORY_AND_LONG_INPUT.md).

## Evidence and licensing

[Typed-output comparison](docs/TYPED_OUTPUTS.md) · [model comparison](docs/COMPARISON.md) · [long-input measurements](docs/MEMORY_AND_LONG_INPUT.md) · [historical research](docs/history/README.md).

Code is Apache-2.0. This kit incorporates unmodified decision, chat and API source from [Eider](https://github.com/rdaum/eider), also under Apache-2.0; our contributions include the bridge, media/transport integration and runtime optimizations. See the [Eider license](licenses/Eider.txt), [attribution and additions](NOTICE), and [pinned source provenance](native/eider-bridge/vendor/PROVENANCE.json). This is an independent project, not an official Eider release. Model and third-party runtime licenses are separate. [License details](docs/LICENSES.md). Research fixtures/results stay in Git, outside installed runtime and source release artifacts. Never treat provisional labels as human-gold certification.
