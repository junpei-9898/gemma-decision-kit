# Memory footprint, Laya input coverage and long-input history

## Memory: measured usage versus minimum capacity

### Model memory footprint

| NVFP4 mode | Peak live GPU tensors | Peak allocator-reserved memory |
|---|---:|---:|
| Optimized text |21.99GiB|23.91GiB|
| Images/video (`--media`) |23.11GiB|24.10GiB|

Measured in the v0.2.0 GB10 regression; media peaks include initialization profiling. Reserved memory **includes** live allocations: do not add the columns. These are PyTorch GPU measurements, excluding some driver/host allocations, not minimum VRAM certifications. GB10 shares system memory between CPU and GPU, so allow additional room for the OS, runtime and loading. The machine's121.6GiB capacity is not the model requirement; conversely, these results do not prove that a24GB discrete GPU can run this package. [Memory definitions and evidence](memory-long-input-results.json).

The numbers come from WI053 final text and media regression completions, using the released NVFP4 recipes, one model at a time and sequential questions. Text uses3GiB FP8 KV; media uses3GiB auto/BF16 KV plus the vision path. `torch.cuda.max_memory_allocated` and `max_memory_reserved` measure different overlapping allocator quantities. They are not summed, do not account for all CUDA context/driver allocations, and do not isolate every host-side peak.

Minimum unified RAM or minimum discrete VRAM was not established by a capacity sweep. A different GPU/runtime, context, concurrency or media shape can change requirements. Startup/weight loading needs headroom; do not treat22GiB live tensors as a22GiB capacity promise. OneGiB is2^30bytes; card capacity labels and usable CUDA capacity can differ. No new purchasing recommendation or hardware support claim is inferred here.

## Laya: which fast results truncate?

| Test | State length / type | Median | Truncated requests | Label agreement |
|---|---|---:|---:|---:|
| Short speed probe |54characters,1question|8.8ms|0/20|20/20 (one repeated question)|
| Medium speed probe |2099characters,1question|21.6ms|5/5|0/5|
| Long speed probe |8108characters,1question|23.1ms|5/5|0/5|
| v1quality corpus |Old80 +additional40|Not a single speed probe|0/120|47/120|
| v2quality corpus |Development60 +same additional40|Not comparable full-input evaluation|100/100|20/60 +14/40, with incomplete instructions|

The short measurement is valid for that short input and is not caused by dropping a long document. Laya is a much smaller encoder-based configuration; the experiment does not isolate architecture as the sole cause of speed. Its shipped max_len1024/head_max_len256 limits cause state truncation in medium/long probes and target-claim truncation in v2. We retained shipped limits, recorded incomplete inputs, and do not present the21.6/23.1ms values as equivalent full-document inference. See [configuration comparison](COMPARISON.md) for prompt/model differences.

## Long-input optimization

### Long-input optimization history

**Historical native-runner measurements, not the public API's input limit.** The released API currently caps each question at8192tokens and rejects the30k/50k-character fixtures below. The research runner allowed65536tokens; long-input public API support has not been released.

| State characters / prompt tokens | Stabilized baseline | Final recipe | Time reduction |
|---|---:|---:|---:|
|10,000 / 7,138|1.563s|1.319s|15.6%|
|30,000 / 21,197|6.569s|4.570s|30.4%|
|50,000 / 35,257|15.248s|8.686s|43.0%|

Same GB10, pinned Gemma4 NVFP4 weights and full input fixtures, one question, loaded/shape-warmed, no prefix-cache reuse,3repeats per length. Includes input preparation + native inference; excludes model load, HTTP and profiling. Each length is one document, not a representative long-document benchmark. Improvement includes attention launch tuning, MoE buffer/copy handling and compact ABC projection.

The baseline is the **stabilized native CUTLASS** recipe. The earlier FlashInfer path was13.714s at50kcharacters (vs final8.686s,36.7% shorter), but had different stability/quality; at10k it was1.242s and faster than final1.319s. We do not claim every input got faster versus every historical recipe. [Stages, quality checks and limits](memory-long-input-results.json).

### Selected measured stages (seconds)

| Stage |10kcharacters|30kcharacters|50kcharacters|
|---|---:|---:|---:|
|WI032 original FlashInfer path|1.242|5.615|13.714|
|WI032 stabilized native CUTLASS +scale correction|1.563|6.569|15.248|
|WI034 query32 attention launch|1.472|5.842|12.692|
|WI037 attention warp/stage tuning +MoE sum-out|1.369|4.713|8.927|
|WI041 final accepted compact-ABC recipe|1.319212|4.570492|8.685709|

Same fixed state fixtures (10000/30000/50000characters;7138/21197/35257prompt tokens), one question,3repeat medians, GB10, NVFP4, FP8 KV3GiB, prefill8192, prefix reuse off. Measurements span separate stage/process runs on2026-09-21–23, not one randomized before/after trial. Do not add marginal gains from separately tested rejected candidates. Final-stage follow-up values, rather than the fastest exploratory probe, are used above. Values from early reports are rounded to milliseconds.

Quality evidence: WI034 kept all88reference probabilities identical to the stabilized baseline (82/88AI-provisional labels). WI041 kept221/221outputs and probabilities identical to its WI039 reference; these221include repeats/warmups, and170/184frozen labeled outputs were correct. This is not a claim of221correct independent questions or universal lossless behavior. Pre-stabilization FlashInfer scored80/88, whereas the stabilized baseline scored82/88; those were different numerical/stability configurations. Do not describe the entire history as one purely speed-only transformation with every probability unchanged.

The50k result is full-prefill latency, **not** the separate historical ~0.17s cached-prefix experiment. Prefix reuse requires prior document processing and storage, and uses different fixtures. Those cache-hit numbers are excluded from this speedup table.

[Machine-readable evidence](memory-long-input-results.json) contains byte-derived memory values, original coverage counts, computed speedup ratios and hashes of internal source reports. The full private corpus remains undistributed. No new inference was run for this documentation update. Public long-input acceptance above8192tokens would require a separate API/configuration change and release validation.

A subsequent context-extension trial is documented [separately](CONTEXT.md). Near64K technical execution succeeded, but the quality gate failed; the public8192input limit remains unchanged.
