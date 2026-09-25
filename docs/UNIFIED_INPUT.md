# Unified local inputs

`analyze` accepts one local source and Eider typed question JSON. It inspects
actual streams, automatically selects text/vision/audio processing and returns processing
coverage. Install [Gemma](INSTALL.md) and [MOSS/FFmpeg](AUDIO.md) first. Weights remain separate.

```sh
gemma-decision analyze --model-path /models/nvfp4 \
  --source /input/recording.mp4 --input examples/request.json \
  --audio-model-path /models/moss --audio-python /state/moss-env/bin/python \
  --cache-dir /state/transcript-cache --output /state/analysis.json
```

No `--media` or `--audio` switch is needed. Text uses UTF-8 `.txt`/`.md`; images are actual
PNG/JPEG; supported audio/video containers are inspected with FFprobe. A video without an
audio track needs no speech model. Audio tracks require MOSS; they are never silently ignored.
Multiple audio/video tracks are rejected rather than guessing the intended language/camera.
An embedded cover picture is not treated as video. Offsets exceeding100ms are currently
rejected: general track retiming is not implemented.

## Processing and evidence

1. Inspect source, duration and planned decision count before loading models.
2. Transcribe audio once with MOSS into anonymous speaker IDs and utterance spans.
3. Exit MOSS, then load one Gemma engine with vision enabled when required.
4. Process every video interval in fixed windows of at most10seconds, including the final
   shorter interval. Resize to fit640×360 and sample2fps before the pinned model's own
   frame selection. Pair each window with overlapping utterances in original source seconds.
   Whole utterances crossing a boundary appear in both windows, explicitly marked; they
   are not split or assigned to a precise word time. Clip-relative time0 maps to window.start.
5. Return each window's provisional choice and its source-time evidence. For multiple windows,
   a final Gemma call considers the local choices and question semantics; no probability
   averaging or majority rule is programmed.

**Multi-window aggregation supports choice questions only; score/noul is rejected before loading models.** Explicit any/all requires three mapped criteria. It is not a visual narrative
summary, and it does not re-examine all original frames/transcripts together. Cross-window
causal relations or ambiguous references can remain unresolved. Use an explicit insufficient-
evidence choice in your question when uncertainty matters. `evidence` lists inspected material;
it is not a claim that every included utterance causally supports the final decision. Sampled
frames can miss brief events. Voice IDs are not mapped to faces or names. If MOSS omits every speaker tag but returns fully
parseable timestamped speech, reserved `S0000` means unknown identity and
`missing_speaker_labels` is reported; it does not certify that only one person spoke. Emotion, music and
other non-speech sounds are not understood by the transcript path.

No question-based skipping is implemented: the full timeline is partitioned and visited, but
that is not full-frame coverage. Static-slide selection and scene-change optimization are
future work. An arbitrary long video is not passed directly to the10second backend.

## Results, limits and failure

Output contains `status`, `source` metadata/SHA256, `coverage`, `evidence`, `decision`,
`decision_scope`, `limitations`, `audio.transcript` and total token/decision usage.
Source names/paths are not returned; transcripts and decisions are still private content.
`--output` creates a new0600file, refusing overwrite. Without it, output goes to stdout.

Maximum source4GiB and30minutes; maximum512planned decisions, configurable lower with
`--max-decisions`. A25minute video with one question plans150window calls plus one final call.
`--max-total-tokens` defaults1,000,000 (maximum4,000,000), enforced before each decision request.
Vision context remains8192tokens, including the final aggregation in that resident engine.
Large questions/transcripts/aggregations may fail rather than truncate. No automatic hierarchy
is claimed beyond this bound. Text uses the existing default context limit.

Preprocessing failures before Gemma produce a nonzero exit without an accepted result. A later
failure returns `failed`/`partial`, no final decision, completed evidence and unprocessed windows;
CLI exits2 and HTTP returns422. A token-limit failure after all windows may have no unprocessed
windows but still no final decision. Do not treat any partial result as an overall answer.
Process-wide initialization failure is fatal. Previously written outputs are never overwritten.

Optional transcript cache requires a private0700directory. Key includes sourceSHA256, pinned
MOSS revision, adapter version and transcription settings. Cached JSON is validated before
reuse, and publication is atomic. It is a trusted local cache, not a signed correctness proof;
do not allow other users to modify it. Frame extraction is regenerated, not cached. Cache is
opt-in because it retains private transcript content. Changing questions reuses transcription.

## Python and loopback HTTP

Python: `gemma_decision.inputs.analyze(body, source, engine_factory=..., audio_model_path=...,
audio_python=...)`. Factory receives `media=True/False`; for memory isolation, do not preload
Gemma before MOSS. A native backend can be loaded only once per process; launch a fresh process
for a new job requiring another mode. CLI does this naturally.

```sh
gemma-decision serve-input --model-path /models/nvfp4 \
  --audio-model-path /models/moss --audio-python /state/moss-env/bin/python \
  --cache-dir /state/transcript-cache --port 8766
```

POST `http://127.0.0.1:8766/v1/analyze` as JSON:

```json
{"request":{"state":"Assess the supplied recording.","questions":{"q":{"type":"choice","instructions":"Does the evidence support the claim?","criteria":{"yes":"Supported","no":"Refuted","unknown":"Insufficient evidence"}}}},"source":{"extension":".txt","base64":"ZXhhbXBsZQ=="}}
```

HTTP body<=8MiB, base64field<=6MiB; larger recordings use local CLI/Python. The extension is
an allowlisted format hint, not a file path; content is inspected. No filesystem paths/URLs,
chunked transfer or upload identifiers are accepted. Requests are serialized and launch a fresh
owned worker process, with3600second timeout, private temporary files and no request logs.
Models reload per request. Use the pinned container with `--network none`; library offline flags
alone are not an OS network boundary. Remote access requires an existing authenticated tunnel.
`serve` and `/v1/decisions` accept the same Eider decision envelope and do not accept audio uploads.

GPU validation results and exclusions are recorded in [UNIFIED_VALIDATION.md](UNIFIED_VALIDATION.md).

## Explicit any/all aggregation

For questions whose meaning is "at least one interval" or "every interval", add an explicit
policy to the unified request. The three labels must match that question's criteria exactly:

```json
"aggregation": {"q": {"operator":"any", "positive":"yes", "negative":"no", "unknown":"unknown"}}
```

`any`: one positive wins; all negative gives negative; otherwise unknown. `all`: one negative
wins; all positive gives positive; otherwise unknown. These operate on provisional local
judgments, not on verified ground truth; sampled frames still limit absence/all claims.
The pipeline processes all intervals even if a witness is already found. No final GPU call is
needed for such a question. Logical answers use `probabilities:null`, never a fabricated
confidence distribution. Mixed requests can use explicit policies for some questions and the
experimental model aggregation for others. `/v1/decisions` accepts the Eider envelope without aggregation policies.

When no policy is provided, the model aggregation remains experimental: a synthetic existence
question was answered incorrectly despite a correct positive window result. Do not use that
mode as a validated cross-window reasoning system. Prefer an explicit policy where it fits;
arbitrary relational questions remain limited by local choice evidence.
