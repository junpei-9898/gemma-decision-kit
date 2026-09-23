# Gemma Decision Kit

Local text decisions with three named choices and an **uncalibrated probability distribution**. No prose answers. Two startup profiles: **speed** (Gemma 4 NVFP4/vLLM) and **memory** (Gemma 4 EXL3 4.10bpw/ExLlamaV3).

Experimental release. Independent implementation inspired by state + typed-question interfaces; not an official Jev clone or drop-in Eider API. No new model training and no weights included.

[日本語](README.ja.md)

## Installation

Clone `https://github.com/junpei-9898/gemma-decision-kit` and enter the repository directory.

See [the pinned GPU installation guide](docs/INSTALL.md). Initial GPU validation targets NVIDIA GB10 / Edge Xpert, Linux ARM64, CUDA 13. Other devices are not certified by these benchmarks. Python-only schema/API tests do not require a GPU:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
gemma-decision info
gemma-decision validate --input examples/request.json
python -m unittest discover -s tests -v
```

## Use

After installing the matching GPU runtime and downloading a pinned model:

```sh
gemma-decision predict --profile speed --model-path /models/nvfp4 --input examples/request.json
gemma-decision serve --profile memory --model-path /models/exl3 --port 8765
curl http://127.0.0.1:8765/v1/decisions \
  -H 'Content-Type: application/json' --data-binary @examples/request.json
```

Only one model is loaded. Stop the process normally and restart with the other profile/model path to switch. This is not zero-latency hot swapping. HTTP binds only to loopback; use an authenticated tunnel for remote use. This small serialized API is not a production Internet gateway.

Input: text `state` and 1–64 named `questions`, each with `type: "choice"`, text `instructions`, and exactly three ordered `criteria`. Criteria order maps to A/B/C and affects the prompt. Output has `answers[id].choice` and `probabilities`, plus profile/model provenance. Probabilities are not calibrated correctness guarantees. [Example](examples/request.json).

No streaming, free-text generation, Boolean/score types, arbitrary choice counts, multimodal input, dynamic GPU batching, tensor parallelism, or LoRA in this release. Inputs over the token limit are rejected, never truncated. Default per-question limit8192tokens; published long-input measurements reached6744tokens. More than one question is processed sequentially to preserve the tested path. The NVFP4 implementation uses one constrained output-token step internally; EXL3 reads decision logits directly.

## Evidence and limitations

[Benchmarks](docs/BENCHMARKS.md) distinguish historical optimization measurements from packaged-release validation. Lower memory does not imply faster inference or identical answers. Changing backend can change decisions. Do not infer general non-inferiority from the fixed provisional-label suite.

## License

Project code: Apache-2.0. Certain ExLlama-derived portions remain MIT with their notice retained. See [NOTICE](NOTICE), [third-party/model licenses](docs/LICENSES.md), and [LICENSE](LICENSE). Model weights and container/runtime dependencies have their own licenses and are obtained separately.
