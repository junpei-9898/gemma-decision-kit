# Benchmark scope and release validation

## Historical native inference (GB10 / Edge Xpert)

Warm preparation + inference + answer extraction; excludes model load, extension build, HTTP and logging. These are pre-packaging native measurements, not an HTTP latency promise. Same frozen Japanese decision inputs, no truncation. Model revisions are in `gemma_decision.profiles`. NVFP4 uses the pinned GB10 image (actual vLLM0.26.1.dev0+gf2654939e.d20260726); EXL3 uses6b84a21 with ARM guards. NVFP4/EXL3 were measured at different times, not simultaneous randomized A/B.

| Input | NVFP4 speed recipe | EXL3 before compact head | EXL3 compact128 |
|---|---:|---:|---:|
| 54 state characters /222 prompt tokens,1question |83.32ms|133.51ms|133.02ms|
| 2099 characters /1878tokens,1question |344.47ms|557.13ms|557.18ms|
| 8108 characters /6744tokens,1question |1251.00ms|2068.96ms|2075.64ms|
| 104 characters,8 independent questions /2360 total tokens,sequential |739.35ms|1237.22ms|1232.35ms|
| Peak PyTorch GPU allocated tensors |21.997GiB|14.463GiB|13.947GiB|

EXL3 medians use10 repeats each; NVFP4 short/eight20, medium/long5. Character count covers state, whereas token count includes question/instructions/template. Memory is not whole-process RSS or a certified minimum VRAM. NVFP4 includes a persistent3GiB FP8 KV allocation; EXL3 is cacheless, so the difference is not attributable to quantization alone. Both use text-only paths. Multimodal memory/quality/speed was not certified here.

Compact EXL3 saves about0.516GiB versus its own prior recipe. Paired full/compact-head measurements showed a few milliseconds saved, but separate end-to-end runs were essentially unchanged. No large speed gain is claimed.

## Provisional quality and rejected settings

On220 fixed AI-provisionally labeled judgments (overlapping120case IDs, including multiple instruction variants), both sequential NVFP4 and EXL3 scored209/220. They did not make identical errors: EXL3 improved one NVFP4 error and lost one NVFP4 correct answer. This is not human-gold certification or a general non-inferiority result.

EXL3 batch2/4 scored210/220 and batch1/8 scored209/220. Numerical confidence differences and a borderline choice change remained. These experimental batch paths are **not enabled in the released API**. The repeated mixed8question workflow was8/8 for EXL3 at every tested batch size (the historical NVFP4 recipe scored7/8). A previous internal EXL3 narrative incorrectly stated7/8; raw predictions and unchanged labels establish8/8.

Host bulk-tokenization/deferred-readback did not give a repeatable speed improvement and is disabled. MoE capacity64/32 was slower. Capacity64 also made an additional mixed-workflow mistake with batch4 (8/8→7/8) and was rejected. Defaults remain capacity128, deterministic fused accumulation, no CPU expert offload, no persistent EXL KV cache, and sequential requests.

## Packaged release checks

The installed package was compared against the same-backend historical predictions: **231/231 choices and all probability values exactly matched for each backend**, including220 fixed quality records and short/medium/long/eight workflows. A real loopback HTTP request was also exercised with the published synthetic example for each backend. This verifies packaging parity on the tested machine, not quality on new data.

[release-validation.json](release-validation.json) records the published audit summary. The historical full fixtures and raw private operational logs are not distributed in this repository; historical accuracy cannot be independently reproduced from the small public example alone. The public contract tests, synthetic request and benchmark command below are reproducible and deliberately do not claim to recreate the historical corpus.

```sh
# After runtime installation: warm measurements on YOUR supplied request.
python3 scripts/benchmark.py --profile memory --model-path /models/exl3 --input examples/request.json --repeats 10
```

Report hardware/runtime/model revision, input token counts, number of questions, warm repetitions and aggregate elapsed time with any comparison. This script reports latency only; it does not invent gold labels or estimate accuracy.
