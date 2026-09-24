# @unit evaluation
# @layer infrastructure
# @work-item-id WI-059
# SPDX-License-Identifier: Apache-2.0
import os
import signal
import subprocess
from .contracts import AudioError

def run_owned(args,timeout,*,env=None,stdout=None):
    if os.name!='posix':raise AudioError('Audio workers require POSIX')
    p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=stdout or subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    try:code=p.wait(timeout=timeout)
    except BaseException as exc:
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait(timeout=5)
        if isinstance(exc,subprocess.TimeoutExpired):raise AudioError('Audio processing timed out; owned worker stopped') from None
        raise
    if code:raise AudioError('Audio subprocess failed; no transcript was accepted')
