# @unit evaluation
# @layer infrastructure
# @work-item-id WI-059
# SPDX-License-Identifier: Apache-2.0
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from .contracts import AudioError,validate_transcript,add_evidence
from .ingest import decode_audio
from .process import run_owned


def write_private_json(path,value):
    """Never overwrite an input, previous transcript, or symlink."""
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
    with os.fdopen(os.open(path,flags,0o600),'w',encoding='utf-8') as out:
        json.dump(value,out,ensure_ascii=False,indent=2);out.write('\n')


def transcribe(source,model_path,*,python=None,max_seconds=1800,timeout=1800,max_new_tokens=16384):
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or not 0<timeout<=7200:raise AudioError('Audio timeout must be in (0, 7200] seconds')
    if type(max_new_tokens)!=int or not 1<=max_new_tokens<=16384:raise AudioError('Audio token limit must be 1..16384')
    if not Path(model_path).is_dir():raise AudioError('A local speech model directory is required')
    with tempfile.TemporaryDirectory(prefix='gemma-audio-') as temp:
        os.chmod(temp,0o700)
        wav,duration=decode_audio(source,temp,max_seconds)
        output=Path(temp)/'transcript.json'
        env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','DO_NOT_TRACK':'1','TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1'}
        # Use exactly this installed package with a separately selected dependency environment.
        root=str(Path(__file__).resolve().parents[2])
        bootstrap='import sys,runpy;sys.path.insert(0,'+repr(root)+');runpy.run_module("gemma_decision.audio.worker",run_name="__main__")'
        args=[python or sys.executable,'-c',bootstrap,'--model',str(Path(model_path).resolve()),'--audio',str(wav),'--output',str(output),'--max-new-tokens',str(max_new_tokens)]
        run_owned(args,timeout,env=env)
        if not output.is_file() or output.stat().st_size>8*1024**2:raise AudioError('Missing or oversized transcript')
        try:result=validate_transcript(json.loads(output.read_text()))
        except (ValueError,TypeError,KeyError):raise AudioError('Worker returned an invalid transcript') from None
        if abs(result['duration_seconds']-duration)>1/16000:raise AudioError('Worker changed audio coverage')
        return result


def predict_audio(body,source,model_path,*,engine_factory,transcript_output=None,**options):
    from ..core import validate
    validate(body)
    if transcript_output is not None and Path(transcript_output).exists():raise AudioError('Transcript output already exists')
    transcript=transcribe(source,model_path,**options)
    prepared=add_evidence(body,transcript)
    if transcript_output is not None:write_private_json(transcript_output,transcript)
    # MOSS has exited before this factory starts Gemma and allocates its model/KV memory.
    engine=engine_factory()
    result=engine.predict(prepared)
    result['audio']={k:transcript[k] for k in ['schema_version','status','model','duration_seconds','sample_rate','warnings','usage']}
    result['audio']['segment_count']=len(transcript['segments'])
    result['audio']['speaker_count']=len({s['speaker'] for s in transcript['segments']})
    return result
