#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# @unit evaluation
# @layer infrastructure
# @work-item-id WI-059
"""Download pinned standalone Linux ARM64 FFmpeg; run without private mounts."""
import argparse
import hashlib
import io
import os
from pathlib import Path
import tarfile
import urllib.request

URL = 'https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz'
SHA = 'f4149bb2b0784e30e99bdda85471c9b5930d3402014e934a5098b41d0f7201b1'
BINARIES = {'ffmpeg': '6bb182d0d75d23028db82e9e4f723ca69b853d055698486e6984ddb2c06fb8ce', 'ffprobe': 'd17ae9b4c297d48e2521ba14e417bb0537c6ff77c584cdbcd6bb0d8d0307a2e8'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination')
    destination = Path(parser.parse_args().destination)
    if destination.exists(): raise SystemExit('Destination already exists')
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = response.read(32 * 1024**2 + 1)
    if hashlib.sha256(archive).hexdigest() != SHA: raise SystemExit('Archive checksum mismatch; no files installed')
    selected = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:xz') as source:
        for member in source.getmembers():
            name = Path(member.name).name
            if name not in {*BINARIES, 'GPLv3.txt', 'readme.txt'}: continue
            if not member.isfile() or name in selected or member.size > 100 * 1024**2: raise SystemExit('Invalid archive member')
            selected[name] = source.extractfile(member).read()
    if set(selected) != {*BINARIES, 'GPLv3.txt', 'readme.txt'}: raise SystemExit('Missing archive files')
    for name, expected in BINARIES.items():
        if hashlib.sha256(selected[name]).hexdigest() != expected: raise SystemExit('Binary checksum mismatch')
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, data in selected.items():
        fd = os.open(destination / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o755 if name in BINARIES else 0o644)
        with os.fdopen(fd, 'wb') as output: output.write(data)
    print('Installed pinned FFmpeg 7.0.2 Linux ARM64 tools')

if __name__ == '__main__': main()
