# License and provenance audit

Checked 2026-09-23. This project distributes adapter code, patches, documentation, a synthetic example and aggregate measurements. It does **not** distribute model weights, CUDA binaries, the runtime container, or Eider's Rust code.

| Component | Examined version/source | License / handling |
|---|---|---|
| Project code | this repository | Apache-2.0; root LICENSE |
| Eider interface inspiration | [bf42ac73](https://github.com/rdaum/eider/tree/bf42ac73bcc0718d9b86cb48cd590e332efd14af) | Apache-2.0; included reference license, no Eider runtime vendored |
| vLLM runtime adaptations | [f2654939e](https://github.com/vllm-project/vllm/tree/f2654939e), packaged version0.26.1.dev0+gf2654939e.d20260726 | Apache-2.0; modifications identified in source/NOTICE; license retained |
| ExLlamaV3 block slicing and ARM patch snippets | [6b84a21](https://github.com/turboderp-org/exllamav3/tree/6b84a21b6f1e5da3f291b9e1019061f0de788279) | MIT; copyright and full permission notice in licenses/ExLlamaV3.txt; these upstream portions remain MIT |
| ARM guard reference | [vcruz305/exllamav3](https://github.com/vcruz305/exllamav3/tree/94ba01d50a13fa9ff672473f2d0eef8b51a71e99) | MIT upstream-derived compatibility changes; no GPU math changes |
| NVFP4 checkpoint | [NVIDIA pinned card](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4/tree/a19cfe00be84568a6867111c9a68c9c44fdcffe6) | card metadata Apache-2.0; downloaded separately |
| EXL3 checkpoint | [turboderp pinned card](https://huggingface.co/turboderp/gemma-4-26B-A4B-it-exl3/tree/7102602329b793263e5490c7a575d5364e6052cb) | card metadata Apache-2.0; downloaded separately |

Google's [Gemma terms page](https://ai.google.dev/gemma/terms) explicitly directs Gemma4 to its [Apache-2.0 license](https://ai.google.dev/gemma/apache_2); do not apply old Gemma terms by assumption. Model-card metadata is recorded in [model-licenses.json](model-licenses.json). No license/NOTICE files were listed in either pinned quantized model's Hub file index at audit time; their Apache references are recorded above. Review the fetched model distribution for changes before redistributing it yourself.

Apache-2.0 is not a requirement that every combined application use the same license. Section4 requires the license copy, retained applicable notices, marked changes and any applicable NOTICE content. We voluntarily use Apache-2.0 for this project's new code and retain the MIT grant for upstream-derived portions. Trademark rights and endorsement are not granted. Eider and vLLM repository root listings had LICENSE but no NOTICE at the audited lookup; this project's NOTICE provides attribution and identifies changes.

The pinned [third-party GB10 image](https://github.com/MiaAI-Lab/Unsloth-Qwen3.6-35b-NVFP4-DGX-Spark) contains separately licensed runtime dependencies (including NVIDIA components). Its image label is not a blanket license for every bundled component. We reference its immutable digest, do not rehost the image, and do not represent its contents as all Apache-2.0. PyTorch, Transformers, Triton, CUDA, marisa-trie and other installed dependencies retain their own notices/licenses in their distributions. The installation guide avoids copying those binaries into this repository.
