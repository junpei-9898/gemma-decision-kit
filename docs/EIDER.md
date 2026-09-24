# Eider semantics with optimized vLLM

The opt-in `--semantics eider` mode uses actual Eider Rust decision preparation and result construction, pinned at `bf42ac73bcc0718d9b86cb48cd590e332efd14af`. vLLM performs Gemma inference. This is a project media/API adapter, not a drop-in replacement for every Eider API.

Build the CPU-only bridge with Rust 1.98.1 (the verified toolchain), a C compiler, and Cargo. Run in the Linux environment/architecture used for inference; a macOS library cannot be loaded on Linux. Cargo dependencies are pinned by Cargo.lock. Building never downloads model weights or installs CUDA. First build needs Cargo dependency access; subsequent cached builds may use `--offline`.

```sh
python3 scripts/build-eider-bridge.py --target-dir "$PWD/state/eider-build"
export GEMMA_EIDER_LIBRARY="$PWD/state/eider-build/release/libeider_decision_bridge.so"
```

For Docker, mount this directory and pass the **container** library path with `-e GEMMA_EIDER_LIBRARY=/state/eider-build/release/libeider_decision_bridge.so`. Build on a compatible Linux/glibc environment. Python wheels contain the Python adapter; the Rust source is in the Git checkout/source distribution. No architecture-specific native binary is silently downloaded.

Use the existing `predict`, `serve`, `analyze`, and `serve-input` commands with explicit `--semantics eider`. The default remains `legacy`: a registered combined audio/video question passed on legacy and failed on Eider, so global non-regression is not established. `--semantics legacy` retains v0.5.0 three-choice prompt/result behavior. Existing choice3 requests remain structurally accepted, but Eider uses a different prompt and may choose a different answer: this is a semantic migration, not numerical equivalence.

The bounded public envelope is `state` (nonempty text), `questions`, optional `model` (`local-gemma4` or the pinned NVIDIA model ID), and optional inline `media`. A different model ID is rejected; the field does not load another model. Eider itself accepts additional state forms; this distribution does not claim those forms. Questions:

- `choice`: 2–64 named criteria, values text or null; named choice and distribution.
- `noul`: probability of true, not a fabricated certainty/binary truth label; optional true/false descriptions.
- `score`: 2–64 ordered text criteria; expected zero-based rubric position and distribution, not just argmax.

```json
{"state":"発送日は金曜日です。","questions":{"friday":{"type":"noul","instructions":"発送日は金曜日ですか？"},"day":{"type":"choice","instructions":"発送日は？","criteria":{"fri":"金曜日","mon":"月曜日"}}}}
```

The wrapper reports `usage.input_tokens` and `physical_input_tokens` as the sum of actual branch prompts, because vLLM processes each question separately. `logical_input_tokens` preserves Eider's shared-prefix accounting for text and is null for media, where the text compiler's temporary marker count is not a meaningful visual-token metric. Use physical tokens to measure visual workload. Probabilities remain uncalibrated.

Images/videos preserve vLLM processor features and expanded tokens. The adapter inserts visual content into Eider's rendered state, verifies text-token roundtrip and the unchanged question suffix, and validates all branch limits before inference. Native Eider is not claimed to encode those images itself.

Audio uses MOSS transcription followed by the same Eider text path. MOSS exits before Gemma loads. Voice tone, environmental sounds, and speaker/face identity are not native Gemma audio judgments. Video uses sampled frames and time-aligned transcript segments; it does not inspect every frame. For multi-window video, score/noul aggregation is explicitly unsupported. Choice supports explicit any/all policies; generic aggregation retains the known cross-window reasoning limitation.

Never infer whole-recording correctness from a successful transport or a small synthetic smoke test. See the release validation report for tested inputs and failures.

Eider's wire fields are preserved: noul returns `{"type":"noul","noul":0.9}`; score returns `score`, a `legend` mapping zero-based string indices to rubric descriptions, and a `probabilities` mapping those indices to numbers. A noul value is a probability, not a JSON Boolean. Client-side thresholding is separate from this API.

```sh
# After building the bridge and setting GEMMA_EIDER_LIBRARY:
gemma-decision predict --semantics eider --model-path /models/nvfp4 --input examples/eider-request.json
gemma-decision serve --semantics eider --model-path /models/nvfp4 --media --port 8765
gemma-decision analyze --semantics eider --model-path /models/nvfp4 --source /input/recording.mp4 --input examples/request.json --audio-model-path /models/moss --audio-python /state/moss-env/bin/python
```
