# Hardware profiles (v0.6.1)

Model selection and hardware tuning are separate. `--profile speed` still selects the one pinned NVFP4 Gemma checkpoint. `--hardware auto|gb10|standard` selects execution policy; `--semantics legacy|eider` independently selects the decision contract. Eider remains opt-in because the AV quality gate is unresolved.

| Policy | Selection / implementation |
|---|---|
| `auto` (default) | Linux ARM64 + device name GB10 + compute capability12.1 → `gb10`; other Blackwell-class CUDA devices → `standard` |
| `gb10` | Rejects other devices. Retains validated Attention launch settings, text MoE reduction/output alias adjustments and compact selected-label text head. Visual mode retains its existing query32 setting and full output head. |
| `standard` | Omits those GB10 runtime/Attention/MoE/head patches. Uses vLLM's ordinary CUTLASS NVFP4 backend and full vocabulary head, retrieving explicit Eider label logits with `logprob_token_ids`. It is a portability baseline, not a speed-equivalent promise. |

Both use the same weights, Eider semantics, sequential scheduling and input limits. Text FP8 KV and visual auto KV settings are retained. Both retain the pinned model's gate/up scale **compatibility correction** (`nv_scale.py`); this is not a GB10 performance tuning option and disabling it would alter weight interpretation. “Standard” does not mean arbitrary vLLM versions, quantizations or models are supported.

GB10 requires the exact documented runtime `0.26.1.dev0+gf2654939e.d20260726`. Standard accepts builds of the same `f2654939e` source revision (version prefix `0.26.1.dev0+gf2654939e`), allowing the build-date suffix to differ. The private preprocessing APIs and scale layout remain pinned. `vllm:latest` is not a supported substitute.

## Other NVIDIA hardware

- Actual GPU validation remains GB10 only. The standard path is tested on GB10 as a control; **RTX5090, RTX PRO Blackwell and B200 deployment remains UNVERIFIED** without those machines.
- The documented GB10 container is ARM64 and must not be presented as an x86 container. Prepare an architecture-compatible runtime from the pinned source with the matching NVIDIA driver/CUDA/CUTLASS support, then install this package and build the CPU Eider bridge for that Linux architecture.
- Do not copy `TORCH_CUDA_ARCH_LIST=12.1a` to a different GPU. Use the build's appropriate architecture configuration or let its toolchain detect the device.
- Pre-Blackwell devices (e.g. RTX4090/3090/H100/A100) fail this recipe's capability check. Alternative quantizations or emulation may be possible elsewhere, but are not part of this supported recipe.
- One GPU only; no tensor parallelism. Required free memory depends on context and media. The measured GB10 allocations do not establish a24GB minimum or prove fit on any particular discrete GPU.

```sh
# Existing GB10 installation; Eider Rust bridge already configured:
gemma-decision serve --semantics eider --hardware auto --model-path /models/nvfp4 --media

# Explicit standard policy in a compatible runtime (other GPU hardware unverified):
gemma-decision predict --semantics eider --hardware standard \
  --model-path /models/nvfp4 --input examples/eider-request.json
```

Python: `EiderEngine('speed', model_dir, hardware='auto')`. The Eider response includes the selected `hardware_profile`. CLI `serve-input` forwards the hardware policy to its isolated workers. Changing profiles requires a process restart; it never changes or downloads model weights.

## GB10 control measurement

Same pinned runtime/weights,64known short Japanese questions (one question/request,121–188tokens, context8192,3GiB KV, sequential). Both modes answered61/64 and all answer/probability fields matched the original WI-075 optimized baseline exactly. Median direct-Python time over63requests after the first: standard84.64ms; auto-selected GB10 mode76.36ms. This small control run establishes preservation on this GB10 suite, not portability to another physical GPU or a universal speed ratio. [Raw comparison summary](../benchmarks/eider-release/hardware-comparison.json).
