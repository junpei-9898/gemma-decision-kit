# Gemma Decision Kit

Local **NVFP4 Gemma 4 decisions for text, images and short videos**: three named choices with an **uncalibrated probability distribution**, without prose generation. v0.2.0 removes the EXL3 backend from the active distribution; historical v0.1.0 remains available.

Experimental release. Independent implementation inspired by state + typed questions; not an official Jev clone or drop-in Eider API. No new model training, no bundled weights. [日本語](README.ja.md).

## Accuracy and speed on our Japanese decision task

**93.3% label agreement (112/120) and ~80ms warm short-question latency on GB10.** This was the highest agreement among the six tested configurations below. In the same-node HTTP comparison, short-question latency was **1.62× faster than DiffusionGemma**, with **+5.0percentage points** higher v1 agreement.

These are fixed Japanese evidence judgments (`supported / refuted / insufficient`) against **AI-provisional labels**, not human-certified accuracy or a general leaderboard. Quality uses120case IDs; latency uses one54-character state repeated20times. Fast answers to that one example do not imply high corpus accuracy.

| Configuration | Agreement (v1,120cases) | Short-question median | Study |
|---|---:|---:|---|
| Gemma Decision Kit · Gemma 4 26B-A4B NVFP4 | **93.3% (112/120)** | **80.4ms** | A |
| DiffusionGemma 26B-A4B NVFP4 | 88.3% (106/120) | 130.6ms | A |
| Eider · Qwen3.6-35B-A3B NVFP4 | 85.8% (103/120) | 250.3ms | B |
| SemIf · Qwen3.5-4B BF16 | 64.2% (77/120) | 119.6ms | B |
| Laya multilingual · 0.322B | 39.2% (47/120) | 8.8ms | B |
| NanoJev · 0.6B navigation checkpoint | 38.3% (46/120) | 40.6ms | B |

**A:** same GB10 node,2026-09-23, loopback HTTP. **B:** earlier2026-09-20 runs on a different GB10 node; Qwen uses HTTP, others native Python APIs. All are loaded/warm medians, excluding startup. Model sizes, quantization, templates and API boundaries differ: this is a configuration comparison, not a controlled model-only speed ranking. Current-kit rows measure its frozen optimized recipe through the evaluation HTTP adapter; v0.2.0 verifies prediction parity, not an identical public-server latency guarantee.

Laya and NanoJev are faster on this short input but have lower label agreement on this task. DiffusionGemma wins the8-question workflow:377.6ms vs732.8ms, with40/40 vs35/40 matching labels across5repeats of the same8questions. The tested NanoJev checkpoint targets navigation, not general Japanese evidence classification. No blanket claim of being faster or more accurate than every OSS model is made.

[Full comparison, instruction sensitivity and immutable versions](docs/COMPARISON.md) · [Machine-readable aggregates](docs/comparison-results.json) · [Text/image/video timings and startup limits](docs/BENCHMARKS.md)

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
