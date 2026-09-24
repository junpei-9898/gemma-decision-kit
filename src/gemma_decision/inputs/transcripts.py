# SPDX-License-Identifier: Apache-2.0
# @unit evaluation
# @layer infrastructure
# @work-item-id WI-063
"""Opt-in private transcript cache with explicit provenance and coverage checks."""
import hashlib
import json
import os
from pathlib import Path
from .. import __version__
from ..audio.contracts import AudioError, manifest, validate_transcript
from ..audio.pipeline import transcribe, write_private_json


def obtain(source, model_path, source_hash, *, cache_dir=None, **options):
    identity = {'source_sha256':source_hash,'model':manifest()['revision'],
                'adapter':__version__,'options':{k:v for k,v in options.items() if k!='python'}}
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    target = None
    if cache_dir:
        root = Path(cache_dir)
        if root.is_symlink(): raise AudioError('Cache directory must not be a symlink')
        root.mkdir(mode=0o700,parents=True,exist_ok=True)
        if root.stat().st_mode & 0o077: raise AudioError('Cache directory must be private (0700)')
        target = root/(key+'.json')
        if target.exists():
            if target.is_symlink() or target.stat().st_size>8*1024**2: raise AudioError('Invalid cached transcript')
            try:
                data = json.loads(target.read_text())
                if data['identity']!=identity: raise ValueError()
                return validate_transcript(data['transcript']), True
            except (ValueError,KeyError,TypeError): raise AudioError('Invalid cached transcript; not reused') from None
    result = transcribe(source, model_path, **options)
    if target:
        temporary = target.with_suffix('.'+os.urandom(8).hex()+'.tmp')
        try:
            write_private_json(temporary,{'identity':identity,'transcript':result})
            os.link(temporary,target)
        finally: temporary.unlink(missing_ok=True)
    return result, False
