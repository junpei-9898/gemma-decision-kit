# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib.util
import json
from pathlib import Path


def verify_exl_source():
    spec=importlib.util.find_spec('exllamav3')
    if spec is None or spec.origin is None:raise RuntimeError('Install the pinned ExLlamaV3 runtime first')
    root=Path(spec.origin).parent
    manifest=json.loads(Path(__file__).with_name('exl_manifest.json').read_text())
    for relative,expected in manifest.items():
        f=root/relative
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Unsupported ExLlamaV3 source: '+relative+'; use scripts/install-memory.sh')
