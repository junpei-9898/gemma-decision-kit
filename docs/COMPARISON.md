# OSS configuration comparison: Japanese evidence decisions

The defensible result is **highest observed v1 agreement among these tested configurations, combined with low single-question latency**. This is not universal dominance in accuracy, throughput, latency, memory or hardware efficiency. No new GPU experiments were run for this documentation update.

### Measured hardware

| Item | Test environment |
|---|---|
| Machine / GPU | Edge Xpert, one NVIDIA GB10 (Blackwell, SM121) per run |
| Memory | CPU/GPU unified memory; OS-visible approximately121.6GiB, **not dedicated model VRAM** |
| Platform | Linux ARM64; kit uses PyTorch2.11.0+cu130; other runtimes pinned per configuration |
| Study-A CPU allocation | Container quota8CPUs, `OMP_NUM_THREADS=4` |

A more powerful GPU **may reduce latency**, especially for compute-heavy long-input prefill, but we have not measured a speedup factor on other GPUs. CPU preprocessing, memory bandwidth and compatible kernels also matter; do not multiply these timings by advertised TOPS or bandwidth ratios. The pinned ARM64/GB10 image is not a validated x86/RTX deployment.

## Dataset, task and denominator

All configurations classify Japanese evidence into supported/refuted/insufficient. The v1 table combines the same old80 and additional40 nonduplicated case IDs:120cases from30source groups, with4correlated variants/group. Labels are frozen AI-provisional judgments, not human gold. Data had already been used during Gemma development: this is regression evidence, not an independent unseen test. Do not treat120case IDs as120statistically independent examples or claim statistical significance.

| Configuration | Agreement (v1,120cases) | Short-question median | Study |
|---|---:|---:|---|
| Gemma Decision Kit · Gemma 4 26B-A4B NVFP4 | **93.3% (112/120)** | **80.4ms** | A |
| DiffusionGemma 26B-A4B NVFP4 | 88.3% (106/120) | 130.6ms | A |
| Eider · Qwen3.6-35B-A3B NVFP4 | 85.8% (103/120) | 250.3ms | B |
| SemIf · Qwen3.5-4B BF16 | 64.2% (77/120) | 119.6ms | B |
| Laya multilingual · 0.322B | 39.2% (47/120) | 8.8ms | B |
| NanoJev · 0.6B navigation checkpoint | 38.3% (46/120) | 40.6ms | B |

The short timing input is54state characters plus the question/choices/template;20warm repeats. It is a separate single-example latency probe, not the average over120quality cases. Short-probe agreement was20/20 for all except NanoJev (0/20). All120v1 quality outputs were present; no failed outputs were removed from the denominator and no v1 quality input truncation was recorded. Ranking configuration-level outcomes is not attributing differences to model architecture alone.

A=WI047,2026-09-23,same GB10 node, loopback HTTP. B=WI017,2026-09-20,another GB10 node, Qwen/Eider HTTP and other candidates native synchronized Python API. Both nodes use Linux ARM64 and CUDA13. Actual optimized-kit vLLM version is0.26.1.dev0+gf2654939e.d20260726; old report shorthand0.26.0 was image branding. No simultaneous A/B randomized trial across all6configurations was performed. Weight precision, model size, tokenizer, native prompt templates and host/runtime versions differ.

Current-kit recipe results come from the frozen evaluation adapter. The distributed v0.2.0 default text path subsequently matched231decisions and their probabilities exactly, including220quality judgments. This bridges predictions, not public-server timing. The old Eider/Gemma adapter scored74/80+38/40=112/120, while the current recipe scores73/80+39/40=112/120; aggregate equality does not mean identical errors. Do not attribute that prompt change to low-level speed patches. A separate same-prompt patch ON/OFF test matched461outputs including warmups/repeats, not461independent examples.

## Same-node DiffusionGemma comparison

All-question completion wall time through loopback HTTP, models loaded, prefix cache disabled, one excluded warmup per condition. Short20repeats; medium/long/eight5repeats. State length excludes instructions/choices, but processing includes them. No input truncation.

|State / questions|Optimized kit recipe|DiffusionGemma|Diffusion time / kit time|
|---|---:|---:|---:|
|54characters /1|80.4ms|130.6ms|1.62×|
|2099characters /1|340.4ms|363.8ms|1.07×|
|8108characters /1|1251.9ms|1359.0ms|1.09×|
|104characters /8|732.8ms|377.6ms|0.52×|

Kit prompt tokens:222/1878/6744; Diffusion:267/1923/6789. Eight-question kit prompts total2360tokens. Kit executes8sequential generation calls. Diffusion groups6+2questions: facade submits groups concurrently, GPU max_num_seqs=1 processes groups one at a time. Within each group it jointly judges multiple questions. Diffusion is1.94× faster on this workflow and matches40/40labels vs kit35/40 across5repeats. These are8unique questions, not40. The kit's mixed-document error reproduces with speed patches both ON and OFF; it is a known input-form weakness, not dismissed as occasional noise.

Single v1 quality: kit112/120 vs Diffusion106/120, a5.0percentage-point difference in this fixed corpus. The medium/long speed differences have only5observations and are not a broad statistical claim. This comparison does not cover every Diffusion setting or simultaneous throughput.

## Instruction sensitivity and candidate-specific limitations

All220per-configuration outputs include repeated case IDs under v1/v2. The main table deliberately uses only120nonduplicated v1case IDs. v2was previously designed using Gemma error analysis, then frozen for other models; it is not neutral per-model tuning.

|Configuration|Old80 v1|Additional40 v1|Same40 v2|Old development60 v2|
|---|---:|---:|---:|---:|
|Current kit|73/80|39/40|40/40|57/60|
|DiffusionGemma|69/80|37/40|36/40|52/60|
|Eider Qwen3.6|68/80|35/40|35/40|53/60|
|SemIf|54/80|23/40|37/40|51/60|
|Laya|33/80|14/40|14/40†|20/60†|
|NanoJev|28/80|18/40|18/40|17/60|

SemIf's40-case agreement improves from57.5% to92.5% with v2; its lower v1 result is not a ceiling on its capability. †Laya truncates the target-claim-containing head in all100v2requests, so those outputs are not equivalent full-input quality comparisons. Its medium/long speed probes also truncate state and were incorrect0/5; those fast times are not advertised as full-document processing. Laya is run with its shipped multilingual settings, not tuned to this task.

NanoJev uses the user-selected root best.safetensors checkpoint, described by its pinned README/run_config as navigation-oriented (coordinate teacher training, max_length512). Evaluation sets the public predictor limit to8192; longer inputs extrapolate beyond that training length. No retrospective checkpoint cherry-picking or Japanese fine-tuning was performed. Its short speed probe was incorrect0/20. These results do not characterize all NanoJev variants or its intended navigation performance.

For measured memory and an explicit short/long Laya coverage table, see [memory and long-input evidence](MEMORY_AND_LONG_INPUT.md).

## Immutable versions and reproduction limits

|Configuration|Source pin|Weights / revision|Precision and route|
|---|---|---|---|
|Kit|[v0.2.0](https://github.com/junpei-9898/gemma-decision-kit/tree/v0.2.0), pinned vLLM f2654939e|nvidia/Gemma-4-26B-A4B-NVFP4 @ a19cfe00be84568a6867111c9a68c9c44fdcffe6|NVFP4, sequential ABC decision, FP8 KV3GiB|
|DiffusionGemma|PR407326b7, wheel48683121, imagefc120ece0a38|nvidia/diffusiongemma-26B-A4B-it-NVFP4 @ ec4ff3df205028f4e81c954c2227f9312b3ec2ea|NVFP4, samples1/steps1/think0/seed42/canvas64, FP8 KV4GiB|
|Eider Qwen|Eider bf42ac73bcc0718d9b86cb48cd590e332efd14af|nvidia/Qwen3.6-35B-A3B-NVFP4 @ 491c2f1ea524c639598bf8fa787a93fed5a6fbce|NVFP4, Eider decision API|
|SemIf|ca3ba65f142967030ecb453346e94d6f476a69df|Qwen/Qwen3.5-4B @ 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a|BF16, direct scorer|
|Laya|d113dca2512fb3eaca313534bc54c7162d87c1d4|convaiinnovations/laya-multilingual @ 052592a15d198d9ad47da779604259b10b47b7aa|0.322B, BF16 autocast, shipped limits|
|NanoJev|71a513bb0163b5634467842b523ee0c0ed6fb1c7|C-Tianyu/NanoJev @ 4a19595eada0857133c0d2be024f879a4077054b|FP32 storage/BF16 autocast, root checkpoint|

SemIf/Laya/NanoJev used torch2.11.0+cu130 and isolated Transformers5.17.0, not necessarily their authors' exact reference environments. Author claims of official Jev equivalence are not evaluated. These are independent OSS configurations; no official Jev API benchmark was performed.

[comparison-results.json](comparison-results.json) exports source-derived counts, all decision latency conditions, truncation counts, model pins and hashes of the internal source reports. The private full corpus, raw prompts/predictions and operational logs are not redistributed. You can check arithmetic, inspect public synthetic examples and run the supplied latency script on your own data; you cannot independently reproduce the full historical accuracy evaluation from this repository alone. Source-report hashes establish provenance, not third-party verification.

For image/video results, memory, startup/JIT limits and the installed-wheel regression see [BENCHMARKS.md](BENCHMARKS.md). There is no cross-OSS multimodal comparison supporting a claim of image/video superiority.
