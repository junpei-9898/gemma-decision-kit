# Local MOSS audio (v0.4.0)

`transcribe` produces Japanese/multilingual text, anonymous speaker labels and utterance times.
`predict --audio` runs this step and supplies the transcript to the existing Gemma choice engine.
MOSS exits before Gemma loads. This adds preprocessing; it does not add an audio encoder to Gemma.
GPU end-to-end acceptance is pending; see [validation status](AUDIO-VALIDATION.md).
Audio is CLI/Python only. Existing HTTP endpoints do not accept audio files or paths.

## Pinned worker setup

Use the exact GB10 image and existing `/state/package` target in [INSTALL.md](INSTALL.md).
Create the worker in a **new** directory. This preserves the runtime's CUDA/PyTorch and avoids
changing Gemma's Transformers environment. Run setup with network, without mounting private audio.

```sh
docker run --name gemma-audio-setup --user "$(id -u):$(id -g)" \
  -e HOME=/state -v "$PWD":/app:ro -v "$PWD/state":/state \
  --entrypoint bash "$IMAGE" -lc '
    test ! -e /state/moss-env &&
    python3 -m venv --system-site-packages --without-pip /state/moss-env &&
    python3 -m pip --python /state/moss-env/bin/python install \
      "transformers==5.8.1" "accelerate==1.14.0" "soundfile==0.13.1" "librosa==0.11.0"
  '
```

The worker needs CUDA PyTorch (tested2.11.0+cu130 in the pinned image), ffmpeg/ffprobe, and
those audio dependencies. The base package remains lightweight; `[audio]` provides the Python
audio dependencies for a **separate** environment, not a recommendation to overwrite Gemma's environment.
The CLI launches its own installed worker code with `--audio-python`, so the two environments
use the same adapter version. Generic GPU support and different runtime versions are unverified.

Download the pinned MOSS weights separately (about1.83GB). The source executes custom model code;
its16 required files are SHA256-verified before loading. The manifest is inside the wheel.
No credentials or gate acceptance are automated.

```sh
docker run --name gemma-audio-download --user "$(id -u):$(id -g)" \
  -e HOME=/state -e PYTHONPATH=/state/package -v "$PWD/state":/state \
  --entrypoint python3 "$IMAGE" -m gemma_decision.cli download-audio \
  --audio-model-path /state/model-moss
```

## Audio decoding tools

The pinned base image does not include ffmpeg/ffprobe. Install these separately into a new
state directory using the script in the source distribution (Linux ARM64 only):

```sh
docker run --name gemma-audio-tools --user "$(id -u):$(id -g)" \
  -v "$PWD/scripts":/scripts:ro -v "$PWD/state":/state \
  --entrypoint python3 "$IMAGE" /scripts/download_audio_tools.py /state/audio-tools
```

Run setup before placing private data in state; do not mount recordings during network-enabled setup.
The script verifies the archive and both binary SHA256 hashes and refuses existing destinations.
It retrieves FFmpeg7.0.2 from [the build provider](https://johnvansickle.com/ffmpeg/), preserving
its GPLv3 license and readme. The binaries are not included in this Python package.
For runtime containers add `-e PATH=/state/audio-tools:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`.
On other platforms supply compatible ffmpeg/ffprobe yourself; those runtimes are unverified.

## Use the same offline runtime as INSTALL.md

Mount the input read-only at `/input`; use `--network none`. Keep `/state` private. Existing CUDA
cache/env arguments from INSTALL.md remain necessary for Gemma. Replace its command with:

```sh
python3 -m gemma_decision.cli transcribe --audio /input/recording.mp4 \
  --audio-model-path /state/model-moss --audio-python /state/moss-env/bin/python \
  --output /state/transcript.json

python3 -m gemma_decision.cli predict --model-path /state/model-nvfp4 \
  --input /app/examples/request.json --audio /input/recording.mp4 \
  --audio-model-path /state/model-moss --audio-python /state/moss-env/bin/python \
  --transcript-output /state/transcript-for-decision.json --output /state/decision.json
```

The `state` field supplies the task context; recognized utterances are appended as quoted JSON evidence.
Questions retain exactly three choices. Each question is tokenized in full before decision inference;
existing context and aggregate token caps apply. No silent truncation, automatic summary or retrieval.
Decision JSON keeps existing fields and adds `audio` metadata/usage, without transcript content.
Transcript JSON contains `segments:[{start,end,speaker,text}]`; times are seconds from the recording start.
Without `--output`, JSON is emitted to stdout. File outputs are mode0600 and refuse overwrite.

Python: `from gemma_decision.audio import transcribe, predict_audio`.
`predict_audio(body, source, model_path, engine_factory=..., python=...)` creates the engine only
after the audio subprocess exits. Do not preload a GPU engine in the factory closure if avoiding
simultaneous residency is required.

## Limits and failure behavior

- Default/max input1800seconds; `--audio-max-seconds` can lower it. File cap4GiB. Decode inspects
  actual sample count, not only metadata, and rejects over-limit input without accepting a truncated result.
- Local WAV/FLAC/MP3/M4A/AAC/OGG/OPUS/MP4/MOV/MKV/WebM only; URLs/playlists are not inputs.
  Audio/video start offsets exceeding100ms are rejected rather than silently shifted.
- Default worker timeout1800seconds (maximum7200 via `--audio-timeout`); up to16384 output tokens
  (`--audio-max-new-tokens`). Decode has a separate180second timeout and probe30seconds.
  Token-limit termination without EOS, malformed/unparsed text, invalid timestamps and partial output fail closed.
- No recognized speech prevents a decision. A successfully parsed transcript is not a correctness guarantee.
- Temporary decoded audio/results use a private directory and are removed after success/failure. Timeout/interrupt
  stops only the owned process group. Model cache/weights are not deleted. Offline library flags are set;
  use `--network none` for an OS-enforced network boundary. No cloud speech service is used.
- Anonymous speaker IDs are local to this recording, not names. Times are utterance spans, not word times.
  Simultaneous-speaker/source separation, face matching, long-audio chunk speaker reconciliation and word alignment
  are not implemented. Recognition mistakes and attribution mistakes remain possible.
- CLI jobs are synchronous. Server-side audio uploads/background jobs are not part of this release.

## Measurement scope

The development comparison on GB10 processed one private24m56s meeting with MOSS:250 utterance spans,
2 anonymous speakers,441.51s inference/preprocessing (load excluded),4.91GiB PyTorch allocated peak,
BF16/SDPA/batch1, no quantization. This is not minimum system memory or an isolated repeated benchmark.
The actual meeting is not distributed. User review selected MOSS after also comparing resegmented Qwen.
No formal Japanese CER/DER or independent quality guarantee is claimed. See [release validation](AUDIO-VALIDATION.md) for package tests and the pending GPU acceptance.
