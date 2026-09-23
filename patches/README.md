# ARM compatibility patches (MIT portions)

Apply in order to ExLlamaV3 6b84a21b6f1e5da3f291b9e1019061f0de788279:
1. arm-compatibility.patch
2. arm-compatibility-r1.patch

Original code copyright (c)2025 Turboderp; full MIT permission notice in ../licenses/ExLlamaV3.txt. Modified here to guard x86 intrinsics on ARM and use the ARM pause instruction. Informed by vcruz305/exllamav3 commit94ba01d50a13fa9ff672473f2d0eef8b51a71e99. CPU offload and native tensor parallelism remain unsupported. These patches do not alter GPU arithmetic. All changes are explicit unified-diff hunks.
