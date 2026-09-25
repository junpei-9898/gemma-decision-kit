#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check release contents without loading CUDA or downloading models."""
import argparse,json,tarfile,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args()
root=Path(args.directory)
wheels=list(root.glob('*.whl'));sources=list(root.glob('*.tar.gz'))
assert len(wheels)==len(sources)==1, 'Expected one wheel and one sdist'
with zipfile.ZipFile(wheels[0]) as f:
    names=f.namelist()
    assert 'gemma_decision/eider_engine.py' in names
    assert 'gemma_decision/model.py' in names
    assert 'gemma_decision/audio/model.json' in names
    assert 'gemma_decision/core.py' not in names
    assert 'gemma_decision/profiles.py' not in names
    assert any(x.endswith('/NOTICE') for x in names)
    assert not any(x.startswith(('benchmarks/','docs/history/')) for x in names)
with tarfile.open(sources[0]) as f:
    names=['/'.join(x.name.split('/')[1:]) for x in f.getmembers()]
    for required in ['native/eider-bridge/Cargo.lock','native/eider-bridge/vendor/PROVENANCE.json','native/eider-bridge/src/lib.rs','scripts/build-eider-bridge.py','docs/MIGRATION.md','NOTICE']:
        assert required in names, required
    assert not any(x.startswith(('benchmarks/','docs/history/')) for x in names)
    assert 'src/gemma_decision/core.py' not in names
    assert not any(x.endswith(('.safetensors','.so','.dylib','.mp3','.wav')) for x in names)
print(json.dumps({'status':'PASS','wheel':wheels[0].name,'sdist':sources[0].name}))
