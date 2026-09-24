#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the pinned CPU-only Eider bridge using an existing Rust toolchain."""
import argparse,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--offline',action='store_true',help='Use already cached Cargo dependencies')
p.add_argument('--target-dir',required=True,help='Dedicated build directory, never cleaned by this script')
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
manifest=root/'native/eider-bridge/Cargo.toml'
if not manifest.is_file():p.error('Run from a source checkout or source distribution containing native/eider-bridge')
env={**os.environ,'CARGO_TARGET_DIR':str(Path(a.target_dir).resolve())}
args=['cargo','build','--locked','--release','--manifest-path',str(manifest),'-p','eider-decision-bridge']
if a.offline:args.append('--offline')
subprocess.run(args,env=env,cwd=manifest.parent,check=True)
print('Set GEMMA_EIDER_LIBRARY to the resulting release/libeider_decision_bridge.so (Linux).')
