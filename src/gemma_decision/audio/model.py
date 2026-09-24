# @unit evaluation
# @layer infrastructure
# @work-item-id WI-059
# SPDX-License-Identifier: Apache-2.0
import hashlib
from pathlib import Path
from .contracts import manifest,AudioError

def verify_model(directory):
    root=Path(directory).resolve()
    for name,expected in manifest()['files'].items():
        path=root/name
        if not path.is_file() or path.stat().st_size!=expected['bytes']:raise AudioError('Speech checkpoint incomplete or not pinned revision')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
        if digest.hexdigest()!=expected['sha256']:raise AudioError('Speech checkpoint hash mismatch')
    return root

def download_model(directory):
    root=Path(directory).expanduser().resolve()
    if root.exists():raise AudioError('Choose a new speech model directory')
    from huggingface_hub import snapshot_download
    pin=manifest()
    snapshot_download(repo_id=pin['id'],revision=pin['revision'],allow_patterns=list(pin['files'])+['README.md'],local_dir=root)
    verify_model(root)
    return {'model':pin['id'],'revision':pin['revision'],'verified':True}
