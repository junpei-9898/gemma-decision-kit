# License and provenance audit

Pinned-source audit 2026-09-23, distribution updated 2026-09-25. This project distributes adapter code, documentation, synthetic examples and aggregate measurements. It does **not** distribute model weights, CUDA binaries, the runtime container. Pinned Eider CPU Rust modules are included under native/eider-bridge/vendor.

| Component | Examined version/source | License / handling |
|---|---|---|
| Project code | this repository | Apache-2.0; root LICENSE |
| Eider CPU decision/chat/API source | [bf42ac73](https://github.com/rdaum/eider/tree/bf42ac73bcc0718d9b86cb48cd590e332efd14af) | Apache-2.0; retained license/notices and provenance; thin ABI/media adapters and standalone build workspace identified as project additions; vendored files unchanged |
| vLLM runtime adaptations | [f2654939e](https://github.com/vllm-project/vllm/tree/f2654939e), packaged version0.26.1.dev0+gf2654939e.d20260726 | Apache-2.0; modifications identified in source/NOTICE; license retained |
| NVFP4 checkpoint | [NVIDIA pinned card](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4/tree/a19cfe00be84568a6867111c9a68c9c44fdcffe6) | card metadata Apache-2.0; downloaded separately |

Google's [Gemma terms page](https://ai.google.dev/gemma/terms) explicitly directs Gemma4 to its [Apache-2.0 license](https://ai.google.dev/gemma/apache_2); do not apply old Gemma terms by assumption. Model-card metadata is recorded in [model-licenses.json](model-licenses.json). No license/NOTICE files were listed in the pinned NVFP4 model's Hub file index at audit time; their Apache references are recorded above. Review the fetched model distribution for changes before redistributing it yourself.

Apache-2.0 is not a requirement that every combined application use the same license. Section4 requires the license copy, retained applicable notices, marked changes and any applicable NOTICE content. We voluntarily use Apache-2.0 for this project's new code. Trademark rights and endorsement are not granted. Eider and vLLM repository root listings had LICENSE but no NOTICE at the audited lookup; this project's NOTICE provides attribution and identifies changes.

The pinned [third-party GB10 image](https://github.com/MiaAI-Lab/Unsloth-Qwen3.6-35b-NVFP4-DGX-Spark) contains separately licensed runtime dependencies (including NVIDIA components). Its image label is not a blanket license for every bundled component. We reference its immutable digest, do not rehost the image, and do not represent its contents as all Apache-2.0. PyTorch, Transformers, Triton, CUDA, Pillow, OpenCV and other installed dependencies retain their own notices/licenses in their distributions. The installation guide avoids copying those binaries into this repository.

v0.2.0 removes ExLlamaV3 code, installer, patch snippets and model profile from the active distribution. v0.1.0 remains a historical tagged release with its original MIT notices. This does not remove or change upstream licensing obligations in that old release. No new upstream license/version is introduced by v0.2.0.

## v0.3.0 context extension

Context configuration, contract tests and the synthetic benchmark/plot scripts are project-authored Apache-2.0 additions. The benchmark facts and background records are fictional project-created examples. No model weights, new upstream code, runtime dependency, model revision or model-license terms are added or changed. Existing LICENSE, NOTICE and upstream obligations remain in force.

## v0.4.0 MOSS audio

The optional MOSS worker adapts the prompt/processing flow and vendors the transcript parser from [OpenMOSS commit61bc29cd](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/tree/61bc29cd4120be7b5d3b761b64cd5dff57263642), Apache-2.0. Its license is included as licenses/Apache-2.0-MOSS.txt, provenance/modifications are recorded in source and NOTICE. Model weights are separately downloaded at704aa4a9c304e8520be88901e0d1960158ef5b15. Fixed hashes in audio/model.json verify the checkpoint including its executable Python files. ffmpeg/PyTorch/CUDA/Transformers remain separately installed dependencies, not bundled binaries. Private meeting materials and locally synthesized validation audio are not distributed.

## Typed-output evaluation artifacts

The new fictional benchmark cases, evaluation bridges, scoring summaries and recorded outputs are project-created evaluation artifacts under Apache-2.0. They import separately provisioned pinned OSS implementations; no upstream weights, dependencies, private user data or additional runtime binaries are included. Existing upstream license obligations remain unchanged.


## Eider attribution verification (2026-09-25)

All 25 files listed in [PROVENANCE.json](../native/eider-bridge/vendor/PROVENANCE.json), including the two vendored Cargo manifests and LICENSE, were byte-compared with the pinned upstream commit and are unchanged. The Eider license copies in `licenses/Eider.txt` and `native/eider-bridge/vendor/LICENSE` match upstream. The pinned upstream tree contains no NOTICE file; our root [NOTICE](../NOTICE) and [bridge NOTICE](../native/eider-bridge/NOTICE) describe attribution and project additions, rather than claiming to reproduce an upstream NOTICE.

The standalone bridge workspace, Rust ABI and Python integration are project additions. We do not claim authorship of the vendored Eider implementation. If those upstream files are changed in a future release, the changed files must carry prominent modification notices; update the provenance record as well. Retain applicable upstream copyright, patent, trademark and attribution notices. See [Apache-2.0 section 4](https://www.apache.org/licenses/LICENSE-2.0#redistribution).

Source distributions include the vendored source, its LICENSE, bridge NOTICE and provenance. Wheels include the project LICENSE/NOTICE and third-party license texts, including Eider; the Eider bridge is built separately from the supplied source. These files describe the shipped code; model weights and third-party runtime images remain separate distributions with their own terms.
