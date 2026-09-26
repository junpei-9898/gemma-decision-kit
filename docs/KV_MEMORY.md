# Spark64K text memory reservation

The default KV reservation is now **1.625GiB for Spark64K text**, including `--hardware auto` when it resolves to `spark`. This reduces spare allocation, not model precision, input content or context length. Text still accepts65,535input tokens per question; no truncation was introduced.

| Configuration | KV reservation |
|---|---:|
| Spark text,64K internal context |1.625GiB|
| Standard text,64K |3GiB|
| Either profile,128K /256K text |4GiB /8GiB|
| Media, including text requests in a media process |3GiB|

Explicit internal backend budget overrides are preserved. The public CLI has no new setting. Weights, FP8 text KV precision, Eider typed semantics and8192-token prefill chunks are unchanged. Smaller text input limits within the64K tier use the same tier budget.

## Measured comparison —2026-09-26

Edge Xpert / NVIDIA GB10, Linux ARM64, one resident model and sequential requests, pinned NVFP4 revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6`, vLLM `0.26.1.dev0+gf2654939e.d20260726`, eager mode. PyTorch peaks include startup profiling. Each length had one warmup then three timed repetitions; timing includes Eider and inference, excludes HTTP/model loading/output recording.

| Measurement | Previous3GiB KV | New1.625GiB KV |
|---|---:|---:|
| Peak allocated |21.998GiB|20.623GiB|
| Peak allocator reserved |24.156GiB|23.047GiB|
|313input tokens, warm median |91.73ms|91.05ms|
|35,113tokens, warm median |8.632s|8.635s|
|64,113tokens, warm median |20.028s|20.014s|

All52outputs had identical answers and candidate logits:40existing AGNews choice regression inputs plus three synthetic length probes repeated four times. These are52outputs, not52independent new accuracy examples. Results support output parity in this scope, not universal accuracy equivalence or statistically established speed improvement. The baseline completed all inference before a shutdown-time process-ownership guard fired; that event was retained, the ownership monitor was corrected, and subsequent trials exited cleanly without swap growth.

Reserved includes allocated: do not add them. These figures exclude some driver/host allocations and do not certify a20.6GiB or24GB GPU. GB10 shares system memory; loading/runtime/OS need additional headroom.128K/256K,media and other GPUs were not newly capacity-tested for this change and retain their previous budgets.

Reducing prefill chunks further changed probabilities and slowed long inputs, so it was not adopted. Bypassing the sampler showed no further peak-memory reduction and was not adopted. No new kernels or prefill engine were introduced.

## Release regression

The actual release source (without a trial KV override) passed62CPU contract tests in the pinned Linux environment and reproduced all52baseline answers/logits above. Six additional synthetic cases—noul,score5,choice2/8/32/64—also matched a separately loaded3GiB reference exactly. These cover the changed default and typed output plumbing; they do not establish accuracy on unseen tasks. CPU configuration tests preserve media, standard hardware,128K/256K and explicit backend overrides. No new image/video/MOSS GPU inference was performed for this text-only change.
