> Historical prototype notes (WI-072). The released API and media migration are documented in [EIDER.md](../EIDER.md).

# Experimental direct Eider decision bridge

This branch is an evaluation candidate, not the default API or a quality-certified release.

```text
typed state/questions
  → Eider DecisionApiRequest::into_prompt_request
  → Eider DecisionPromptCompiler::prepare
  → serialized vLLM selected-logit inference
  → Eider answers_from_logits (temperature 1)
  → Eider DecisionApiResponse::from_completion (uncalibrated)
```

`native/eider-bridge` embeds unmodified CPU runtime/format crates and original API
protocol/decision modules from rdaum/eider commit
`bf42ac73bcc0718d9b86cb48cd590e332efd14af`. SHA-256 provenance and Apache-2.0
license are in `vendor/`. This avoids linking the native GPU inference crate.
The new C ABI and Python ctypes transport do not implement decision semantics.
No network service or subprocess is added to the per-request application path.

Build the library with `cargo build --release --locked` in `native/eider-bridge`.
Use an isolated Rust installation and the recorded lockfile. GB10/ARM64 is the
only integration target tested in WI-072; other hosts need their own build/test.

```python
from gemma_decision.backends import SpeedBackend
from gemma_decision.eider_direct import EiderBridge, EiderDecision
backend = SpeedBackend(model_dir, context=8192, decision_logits=True)
bridge = EiderBridge(library_path, model_dir)
decision = EiderDecision(bridge, backend)
result = decision.predict({"model": "local-gemma4", "state": "...", "questions": questions})
```

The backend uses pinned vLLM `raw_logits` mode. Selected values are taken after
the model's softcap and before sampler penalties/temperature/softmax. The
logprob-named output field contains logits in this mode; it is not exponentiated.
Missing/nonfinite values are errors. Eider performs softmax once. No calibration
artifact is applied. Score is Eider's expected zero-based rubric index, not an
arbitrary normalized scalar. The original API shape is not silently replaced.

One model/backend/bridge per process. Entire requests are serialized, including
candidate-head updates and all question branches. The bridge retains one prepared
native request until finish/discard. Eider caps questions and options at 64;
the adapter checks every physical branch against the context limit before GPU
execution. No truncation. Logical input usage and repeated physical branch tokens
are separate; this adapter does not implement shared-prefix branching.

Cancellation during a synchronous GPU call is not a supported public contract.
WI-072's external supervisor controls evaluation timeout and checks cleanup.
No throughput/concurrency claim follows from request serialization.

Images/video require a separate Eider media adapter; pinned Eider preparation
rejects them. Existing visual and MOSS transcription APIs remain available on
the old path. Media migration, long-video quality, calibrated inference and a
public default switch are not completed by this text bridge.

Rollback: continue using the unchanged public v0.5.0 checkout. Do not enable
`decision_logits` on the existing probability API. Model weights are not included
and retain their own licensing terms.
