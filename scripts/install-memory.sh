#!/usr/bin/env bash
set -euo pipefail
# Run inside the documented isolated Python environment. No sudo/global CUDA changes.
repo_root=$(cd "$(dirname "$0")/.." && pwd)
runtime_dir=${1:?Usage: scripts/install-memory.sh NEW_RUNTIME_DIRECTORY NEW_DEPENDENCY_DIRECTORY}
dependency_dir=${2:?Choose a new isolated dependency directory}
if [ -e "$dependency_dir" ]; then echo "Refusing to overwrite dependency directory" >&2; exit 1; fi
if [ -e "$runtime_dir" ]; then echo 'Refusing to overwrite existing runtime directory' >&2; exit 1; fi
git clone --filter=blob:none --no-checkout https://github.com/turboderp-org/exllamav3.git "$runtime_dir"
git -C "$runtime_dir" checkout --detach 6b84a21b6f1e5da3f291b9e1019061f0de788279
git -C "$runtime_dir" apply --check "$repo_root/patches/arm-compatibility.patch"
git -C "$runtime_dir" apply "$repo_root/patches/arm-compatibility.patch"
git -C "$runtime_dir" apply --check "$repo_root/patches/arm-compatibility-r1.patch"
git -C "$runtime_dir" apply "$repo_root/patches/arm-compatibility-r1.patch"
python3 -m pip install --no-deps --target "$dependency_dir" marisa-trie==1.3.1
PYTHONPATH="$repo_root/src:$runtime_dir:$dependency_dir" python3 -c 'from gemma_decision.runtime_check import verify_exl_source; verify_exl_source()'
