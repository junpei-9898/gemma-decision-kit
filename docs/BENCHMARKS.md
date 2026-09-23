# Benchmark scope and validation

[Cross-OSS Japanese accuracy/latency comparison](COMPARISON.md) · [Source-derived aggregate data](comparison-results.json)

## Historical optimized text path

GB10 / Edge Xpert, Linux ARM64, NVFP4 Gemma4 26B-A4B, pinned runtime/revision in `gemma_decision.profiles`. Warm preparation + inference + extraction, excluding model load, HTTP and logging. This is the default text-only recipe, not a speed promise for `--media`.

| State length / total prompt tokens / questions | Median |
|---|---:|
|54characters /222tokens /1question|83.32ms (20repeats)|
|2099characters /1878tokens /1question|344.47ms (5repeats)|
|8108characters /6744tokens /1question|1251.00ms (5repeats)|
|104characters /2360tokens /8sequential questions|739.35ms (20repeats)|

Historical peak PyTorch allocated GPU tensors:21.997GiB with3GiB FP8 KV pool. This is not whole-process RSS or minimum discrete VRAM certification. On220 AI-provisionally labeled judgments (120overlapping case IDs including instruction variants), the sequential recipe scored209/220. No human-gold or general non-inferiority certification. The full historical private corpus is not distributed, so these historical scores cannot be reproduced from the public examples alone. The former EXL3 comparison remains in the v0.1.0 tag; EXL3 is no longer an active backend.

## v0.2.0 image/video and release checks

Release regression: **231/231 text decisions matched prior choices and probabilities exactly**. The frozen media evidence prompt matched historical expanded token IDs and probabilities on39/39outputs. The generic media API matched a direct native reference in the same cache state on39/39outputs. All13unique questions were correct against frozen AI-provisional labels (9image +4video), with two repeated calls per question also correct.

| New generic media API | Cache-cold median | First identical repeat | Second identical repeat |
|---|---:|---:|---:|
|Image (9questions,368–372tokens)|127.44ms|60.84ms|59.38ms|
|Video (4questions,405–410tokens)|145.06ms|69.02ms|68.79ms|

These times include validation, decode/processor work, inference and extraction inside `predict()`. The data URL already exists; file reading/base64 creation, HTTP and model load are excluded. The runtime/kernels were warm. Cold clears prefix and processor caches. These are medians across different questions, not many independent repetitions of the same prompt. Do not compare directly with historical different prompts and claim a new speedup.

Media peak PyTorch allocated memory including startup profiling:23.110GiB; allocated at trial end21.797GiB, reserved24.095GiB. Text peak21.989GiB in the new regression. GB10 uses unified memory; these do not certify minimum discrete VRAM.

An initial strict comparison mixed cold reference with warm repeated outputs and reported FAIL (max probability difference0.000287, no changed choices). The retained corrected trial compares each matching cache phase and achieves exact parity; cross-cache bitwise identity is not claimed.

See [release-validation.json](release-validation.json) for measured results. Images and videos use their own validated recipe: query32 attention, scale reconciliation, BF16/auto KV,3GiB KV,16384 internal context and bounded prefix/processor caches. Text-only compact-head/MoE changes are not enabled here.

Media fixtures: three synthetic960x640 images (9questions) and two640x360,4-second,4-frame/1fps clips (4questions). The frozen evidence prompt and the new generic API prompt are evaluated separately. Report13unique questions rather than inflating the sample size with repeats. Provisional labels were frozen before predictions. These tests do not certify real-world OCR, long or dense video, or audio.

Video is sampled-frame visual analysis. Cache-cold means prefix and processor caches reset, not model-loading/OS/kernel cold. Repeated calls reuse caches and can change probability values slightly; compare against an equally warmed reference. Startup/first-ever kernel latency is separate. Peak memory includes initialization profiling when noted.

## Reproduce on your data

```sh
python3 scripts/benchmark.py --profile speed --model-path /models/nvfp4 --input examples/request.json --repeats 10
python3 scripts/benchmark.py --profile speed --media --model-path /models/nvfp4 --input examples/image-request.json --repeats 10
python3 scripts/benchmark.py --profile speed --media --model-path /models/nvfp4 --input examples/video-request.json --repeats 10
```

The script reports warm request latency including validation/preprocessing and answer extraction, after one warmup request. In media mode this is repeated identical input with prefix/processor caches enabled. It does not include HTTP/model load, invent labels, or estimate accuracy. Report hardware, immutable runtime/model version, mode, input tokens, media dimensions/frame sampling, question count, repetitions and cache state with comparisons.

Final distributed wheel: clean install and17CPU tests PASS; PNG/JPEG/MP4 examples and text-in-media-mode smoke PASS; four corrupt/over-limit media inputs rejected both by the decoder check and HTTP400; expanded-token overflow rejected. A fresh process still incurs first-use compilation: image example took 19.53seconds after model load (not127ms). The subsequent video example took 0.30seconds. Warm table values are not first-request guarantees.
