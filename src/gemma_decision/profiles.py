# SPDX-License-Identifier: Apache-2.0
PROFILES = {
 "speed": {"backend": "vllm", "model": "nvidia/Gemma-4-26B-A4B-NVFP4", "revision": "a19cfe00be84568a6867111c9a68c9c44fdcffe6", "runtime": "vllm==0.26.1.dev0+gf2654939e.d20260726"},
 "memory": {"backend": "exllamav3", "model": "turboderp/gemma-4-26B-A4B-it-exl3", "revision": "7102602329b793263e5490c7a575d5364e6052cb", "runtime": "6b84a21b6f1e5da3f291b9e1019061f0de788279"}
}
