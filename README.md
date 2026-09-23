# Gemma Decision Kit

Local **NVFP4 Gemma 4 decisions for text, images and short videos**: three named choices with an **uncalibrated probability distribution**, without prose generation. v0.2.0 removes the EXL3 backend from the active distribution; historical v0.1.0 remains available.

Experimental release. Independent implementation inspired by state + typed questions; not an official Jev clone or drop-in Eider API. No new model training, no bundled weights. [日本語](README.ja.md).

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

**Historical native-runner measurements, not the public API's input limit.** The released API currently caps each question at8192tokens and rejects the30k/50k-character fixtures below. The research runner allowed65536tokens; long-input public API support has not been released. A subsequent near64K public-API candidate processed full input in20.843s but failed its synthetic quality gate; the8192limit is retained. [Trial results and native model limits](docs/CONTEXT.md).

| State characters / prompt tokens | Stabilized baseline | Final recipe | Time reduction |
|---|---:|---:|---:|
|10,000 / 7,138|1.563s|1.319s|15.6%|
|30,000 / 21,197|6.569s|4.570s|30.4%|
|50,000 / 35,257|15.248s|8.686s|43.0%|

Same GB10, pinned Gemma4 NVFP4 weights and full input fixtures, one question, loaded/shape-warmed, no prefix-cache reuse,3repeats per length. Includes input preparation + native inference; excludes model load, HTTP and profiling. Each length is one document, not a representative long-document benchmark. Improvement includes attention launch tuning, MoE buffer/copy handling and compact ABC projection.

The baseline is the **stabilized native CUTLASS** recipe. The earlier FlashInfer path was13.714s at50kcharacters (vs final8.686s,36.7% shorter), but had different stability/quality; at10k it was1.242s and faster than final1.319s. We do not claim every input got faster versus every historical recipe. [Stages, quality checks and limits](docs/MEMORY_AND_LONG_INPUT.md).

## Installation

Clone this repository, then follow [the pinned GPU installation guide](docs/INSTALL.md). Validated on NVIDIA GB10 / Edge Xpert, Linux ARM64, CUDA13. Other GPU architectures are unverified. CPU contract tests alone do not certify GPU inference.

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

Input: text `state`, 1–64 named `questions` (`type: "choice"`, `instructions`, exactly three ordered `criteria`), and optionally one `media` object. Choices map to A/B/C in insertion order. [Media contract and limits](docs/MEDIA.md). All questions run sequentially. Inputs over the expanded token limit (default8192 per question) are rejected without truncation. Output includes choice, probability distribution, token usage and media metadata when present. Probabilities are not calibrated correctness guarantees.

HTTP binds only127.0.0.1. Use an authenticated tunnel for remote use. This is a small local serialized API, not an Internet production gateway. No audio, free-text generation, Boolean/score types, arbitrary choice counts, dynamic GPU batching, tensor parallelism or LoRA. Video is sampled-frame visual analysis; its audio track is not processed.

## Evidence and license

[Benchmarks and validation](docs/BENCHMARKS.md) separate native historical speed from packaged API timings. Small synthetic media fixtures do not certify real-world OCR or arbitrary video accuracy.

Code and project-authored examples: Apache-2.0. See [NOTICE](NOTICE), [license audit](docs/LICENSES.md), [LICENSE](LICENSE). Model weights and runtime/container dependencies are obtained separately under their own licenses.
