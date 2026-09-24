# Unified input validation — 2026-09-24

Scope: pinned GB10/LinuxARM64 runtime, Gemma26B-A4B NVFP4 and MOSS BF16/SDPA, one job at
a time within this task. Foreign workloads may overlap; these are functional development
checks, not isolated or repeated speed benchmarks. No formal CER/DER or broad audiovisual
quality certification is claimed. Private recordings are not distributed.

| Check | Result | Meaning |
|---|---|---|
| CPU tests | PASS,47 | Includes actual FFmpeg decoding/windows, timestamp overlap, complete timeline partition, image routing, cache, any/all, pre-inference token budget, partial failure and loopback HTTP validation; decision engine mocked |
| Initial installed-wheel audiovisual trial | FAIL | MOSS omitted speaker tags; strict parser rejected complete timestamped text before Gemma |
| Diagnostic repeat | FAIL | Same missing-label format; failure preserved, no new swap |
| Repaired MOSS + Gemma CLI,12s synthetic MP4 | Technical PASS; quality FAIL | Two windows completed; first local answer yes, second unknown; experimental final model aggregation incorrectly returned unknown for authored expected yes |
| HTTP same12s source with explicit any policy | PASS | HTTP200, two windows, final yes; actual Gemma, verified MOSS transcript cache reused |
| Real private24m56s recording preprocessing | PASS |1496.625s container duration, audio detected, all150visual windows extracted;83.794s CPU wall. No new transcription or Gemma decisions in this check |
| Legacy `predict --audio` regression | PASS | Actual MOSS then text-mode Gemma, authored Friday answer matched;178input tokens,1decision,161.747s including startup |
| Entire private recording through Gemma | SKIPPED | Would exceed existing6decision GPU test cap; expansion requested but not authorized at report time |

The repaired CLI trial used2600input tokens and3Gemma decisions,194.931s job wall including
startup. MOSS used240prompt/43generated tokens and43forwards,1.192s generation,1,869,502,976
bytes peak live allocation. HTTP used2314input tokens and2Gemma decisions,171.422s job wall
including worker/model startup; its transcript was a cache hit, not a new ASR inference.
These are one-run development probes, not latency guarantees. The explicitly quantified
HTTP check reuses the development fixture; it is not an independent held-out accuracy test.

The previous v0.4.0 smoke was stopped on host swapout. New96GiB cgroup trials retain the same
host memory/swap guard; successful audiovisual trials observed no increase from pswpout6.
This does not establish that the earlier48GiB cap caused the swap event. No foreign service
was stopped. Old failed trials are retained.

`S0000` with `missing_speaker_labels` records genuinely absent speaker tags; it is not a
fabricated speaker identity. Malformed or partly parsed text is still rejected. Generic
multi-window model aggregation remains experimental and has a known failure. Explicit any/all
only makes aggregation logical; it does not make local model predictions or sampled visual
coverage infallible. Speaker-to-face matching and unrestricted cross-window semantic reasoning
are not implemented.

Legacy regression MOSS:190prompt/47generated tokens,47forwards,1.214s generation,
1,869,398,528bytes peak live allocation. Combined GPU decision tests used6decisions and5092
Gemma input tokens; HTTP transport requests are counted separately (one analyze request).
Final installed-wheel CPU tests:47PASS; final runtime26files match the HTTP-tested artifact by SHA256.
