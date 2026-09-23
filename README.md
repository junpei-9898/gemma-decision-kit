# Gemma Decision Kit

Local **NVFP4 Gemma 4 decisions for text, images and short videos**: three named choices with an **uncalibrated probability distribution**, without prose generation. v0.2.0 removes the EXL3 backend from the active distribution; historical v0.1.0 remains available.

Experimental release. Independent implementation inspired by state + typed questions; not an official Jev clone or drop-in Eider API. No new model training, no bundled weights. [日本語](README.ja.md).

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
