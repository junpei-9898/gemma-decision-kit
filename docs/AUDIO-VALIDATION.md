# Audio validation — 2026-09-24

Release candidate 0.4.0 implements MOSS transcription and sequential Gemma CLI integration.
**End-to-end GPU acceptance is UNVERIFIED.** Do not interpret the CPU integration tests as
proof of successful Gemma inference from recorded audio.

- PASS: 37 CPU tests, including existing text/media contract tests, strict transcript parsing,
  private output, real FFmpeg decoding, timeout cleanup, separate worker protocol and
  the audio-to-engine handoff using a test engine. Repeated from an installed wheel.
- PASS: installed candidate wheel on GB10, BF16/SDPA/batch1, pinned MOSS revision;
  one synthetic Japanese clip, 8.207375 seconds, produced one valid utterance.
  190 prompt tokens,47 generated tokens/forwards,1.524s generation,1,869,398,528 bytes
  PyTorch allocated peak. One run; model loading excluded from generation timing.
- FAIL: the same actual `predict --audio` trial was stopped during the following Gemma
  startup by the configured no-new-swapout guard (host counter increased by one page).
  It exited130 after35.352s; no Gemma decision was produced. OOMKilled was false.
  The counter is host-wide; this does not establish which process caused swapout.
- SKIPPED: full private meeting through the packaged Gemma pipeline, because the guard fired.
  Previous MOSS meeting results in AUDIO.md used the comparison runner, not this new package.

The stopped container and private evidence were retained. No unrelated container was stopped.
A new guarded trial after investigating host swap activity is required for GPU acceptance.
No model weights, recordings, transcripts or private logs are included in the distribution.
