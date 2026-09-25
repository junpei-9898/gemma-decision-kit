# SPDX-License-Identifier: Apache-2.0
"""Hardware policy only; no inference or model semantics live here."""
PINNED_VLLM='0.26.1.dev0+gf2654939e.d20260726'
SOURCE_PREFIX='0.26.1.dev0+gf2654939e'


def select_hardware(requested, *, capability, name, machine, system):
    if requested not in ('auto','spark','standard'):raise ValueError('Unknown hardware profile')
    if system!='Linux':raise ValueError('This CUDA distribution requires Linux')
    if len(capability)!=2 or capability[0]<10:
        raise ValueError('This NVFP4 recipe targets Blackwell-class CUDA GPUs; older GPUs need a separately validated backend/quantization')
    gb10=tuple(capability)==(12,1) and 'GB10' in name.upper() and machine in ('aarch64','arm64')
    if requested=='spark' and not gb10:raise ValueError('Spark tuning requires verified GB10 SM121 / Linux ARM64')
    return ('spark' if gb10 else 'standard') if requested=='auto' else requested


def validate_runtime(version, hardware):
    supported=version==PINNED_VLLM if hardware=='spark' else (version==SOURCE_PREFIX or version.startswith(SOURCE_PREFIX+'.'))
    if not supported:raise RuntimeError('Use the pinned vLLM f2654939e runtime; Spark tuning requires the documented exact build. Other versions are unverified.')


def detect_hardware(requested):
    import platform,torch
    if not torch.cuda.is_available():raise RuntimeError('A supported NVIDIA CUDA GPU is required')
    return select_hardware(requested,capability=torch.cuda.get_device_capability(),name=torch.cuda.get_device_name(),machine=platform.machine(),system=platform.system())
