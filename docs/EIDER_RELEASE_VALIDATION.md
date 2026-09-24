# v0.6.0 Eider + vLLM multimodal release validation

2026-09-25, WI-076. Original optimized vLLM router weights retained. Actual Eider CPU prepare/finish, pinned source `bf42ac73bcc0718d9b86cb48cd590e332efd14af`; NVIDIA Gemma4-26B-A4B NVFP4 revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6`; vLLM `0.26.1.dev0+gf2654939e.d20260726`. One Edge Xpert GB10, Linux ARM64, one GPU, sequential questions, 8 CPU container limit, OMP/Rayon4, 3GiB KV. Text test context8192; media internal context16384 and input limit8192. No router re-quantization or model training.

## Measured scope

| Check | Observed result | Timing boundary |
|---|---|---|
| Known 64-case Japanese text regression | 61/64; all answers **and probabilities** identical to WI-075 | 75.94ms median over63 requests after the first; one question,121–188 tokens; resident model, direct Python |
| Authored red/blue/green circle images | 9/9 initial judgments;9/9 replay; choice/noul/score | 148.30ms median of3 replay **three-question workflows**;1086 total physical tokens/workflow; resident model and warm prefix/processor caches |
| Authored3-second circle video | 3/3 initial judgments and3/3 replay | 174.60ms for one replay **three-question workflow**;1623 total physical tokens; resident model and warm caches |
| Synthetic speech → MOSS → Eider | 1/1 expected choice; full transcript processed | 180.32s including cold loading; MOSS generation1.286s,47 forwards; not a resident audio service benchmark |
|12-second moving-shape video + speech,2 windows | Pipeline complete,4 local decisions; speech-only global question correct; combined spatial/speech question returned `unknown` instead of registered `yes` | 217.71s including cold ASR/Gemma loading;4768physical input tokens; no warm latency claim |

AV physical input tokens:4768. MOSS generation1.151s,43 forwards. Every window was processed. Explicit `any` aggregation was used. The registered combined answer remains **FAIL**; it is not discarded or relabeled. The moving circle appears on both left and right, while the original answer criteria permit overlapping interpretations. This fixture cannot by itself certify an accuracy regression or non-regression. The separate video-only spatial diagnostic returned `yes`; that is diagnostic evidence, not a new held-out quality score. Generic long-video aggregation's older known failure remains unresolved.

Image and video replay choices/thresholded noul/score argmax stayed unchanged. Floating probabilities and expected scores did not remain bit-identical across cold and cached requests (images: maximum scalar difference about2.6e-6). Score correctness above uses the most probable rubric level, not exact equality of its continuous expected value.

These are small correlated synthetic integration checks with provisional labels, not human-gold certification or broad multimodal quality claims. The64text cases are a regression set already used in WI-075, not64new independent tests. Failed compatibility trials are retained: initial image processor marker mismatch, then missing explicitly selected raw logits; repaired by using the checkpoint's visual markers and vLLM's `logprob_token_ids`. An initial package smoke assertion used an incorrect expected wire field (`probability` instead of Eider's `noul`); application output was valid and the test was corrected. CPU suite initially failed because its mock executable was on a noexec temporary filesystem; the isolated test mount was corrected.

## Distribution checks

- Python wheel and source distribution build/install;54 installed-package CPU tests passed.
- Rust workspace:85 tests passed,11 environment-dependent tests ignored in the generic suite. The local Gemma upstream-versus-FFI test was separately enabled and passed across64cases.
- Installed CLI validation and loopback HTTP typed text, image input and64-option output checked; unknown model ID rejected. A spatial-video diagnostic was also recorded.
- Vendored file hashes match Eider provenance. Apache licenses/notices retained and project modifications identified. Model weights, CUDA/runtime images, private recordings and native binaries are not redistributed.

The Python wheel requires a separately built CPU Rust bridge. Git/source distribution includes its sources, lockfile, toolchain pin and build script. Other GPU hardware is not established by these GB10 results.

## Memory and operational limits

Across these media/audio runs, PyTorch peak allocated memory reached about23.5GiB and reserved memory about25.4GiB. MOSS's measured peak was about1.74GiB and it exited before Gemma loaded. These are measured allocations for these shapes/configurations, not a minimum VRAM guarantee;24GB compatibility is not proven. CPU memory, driver allocations, input length and other applications also matter.

`serve` keeps the decision model resident. `serve-input`/`analyze` preserve the existing sequential ASR→Gemma lifecycle and cold loading; their end-to-end audio/video time must not be advertised as the short text/image inference latency. Videos are sampled, not exhaustively inspected. Cross-window score/noul aggregation is unsupported and rejected before model loading. There is no accuracy certification for arbitrary recordings.

## Same-fixture AV comparator and default decision

The legacy path was rerun on the same source, questions, cached transcript and explicit `any` policy. It returned both registered answers (`yes`),2/2; Eider returned1/2. The ambiguous moving-shape criteria limit general conclusions, but this does **not** pass a no-regression gate. Therefore Eider is explicitly opt-in (`--semantics eider`) and `legacy` remains the default for existing callers. The new multimodal Eider implementation is available for evaluation; combined AV reliability is an unresolved quality issue. No weights or prompt text were tuned to this failure.
