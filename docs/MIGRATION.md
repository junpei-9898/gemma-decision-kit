# v0.7: Eider-only distribution

This is a breaking simplification, not a new model or an accuracy claim. Eider preparation/finish, model weights, visual adapter and Spark kernel code are unchanged from the previously opt-in Eider path. The combined AV test discrepancy remains unresolved.

| Before | v0.7 |
|---|---|
| `--semantics eider` | Remove flag; Eider is always used |
| `--semantics legacy` / `DecisionEngine` | Removed; old versions remain in Git tags for historical reproduction |
| `--profile speed` | Remove flag; the pinned NVFP4 model is the only model |
| `--hardware gb10` | `--hardware spark`, or use default `auto` |
| `EiderEngine("speed", model_dir, ...)` | `EiderEngine(model_dir, ...)` |
| `predict --audio file` / `predict_audio(...)` | `analyze --source file` / `inputs.analyze(...)` |
| `--transcript-output` | `analyze` output contains `audio.transcript`, or use standalone `transcribe --output` |
| `scripts/download-model.py speed directory` | `scripts/download-model.py directory` |
| Health response `profile` | `semantics: "eider-decision-v1"` |

Build the mandatory CPU Rust bridge and configure `GEMMA_EIDER_LIBRARY` before every decision process. There is no silent legacy fallback. `validate` checks the envelope without GPU/bridge loading; model-specific tokenization is checked during inference.

Clients must follow the [Eider response contract](EIDER.md), including `noul` probability and score legend fields. Three-choice inputs still fit the schema, but old legacy predictions are not promised. `analyze` returns a workflow envelope with its final answer under `decision`; it is not wire-compatible with removed `predict --audio`. File decisions include coverage and private transcript evidence; handle outputs accordingly.

The Python file-input and HTTP validators default to Eider as well. Multi-window score/noul remains unsupported. `serve` stays resident, while `serve-input` uses isolated per-request workers; do not confuse their cold latency or merge their lifecycle. Speech is not a native Gemma audio input.

Research/old response fixtures remain in Git history or `docs/history`/`benchmarks`, not runtime packages. v0.7 CPU/package checks do not constitute fresh GPU quality or speed measurement. Model adoption remains a user decision.
