"""Explicit model download; review upstream license/card first. Never deletes files."""
import argparse
from pathlib import Path
from gemma_decision.profiles import PROFILES
p=argparse.ArgumentParser();p.add_argument('profile',choices=PROFILES);p.add_argument('directory');a=p.parse_args()
if Path(a.directory).exists():p.error('Choose a new directory; refusing to mix checkpoints')
from huggingface_hub import snapshot_download
m=PROFILES[a.profile]
snapshot_download(repo_id=m['model'],revision=m['revision'],local_dir=a.directory)
