# @unit evaluation
# @layer infrastructure
# @work-item-id WI-059
# SPDX-License-Identifier: Apache-2.0
import json
import math
import re
from importlib.resources import files

class AudioError(ValueError):
    """Errors must never include recognized content or private input paths."""

def manifest():
    return json.loads(files(__package__).joinpath('model.json').read_text())

def validate_transcript(value):
    required={'schema_version','status','model','duration_seconds','sample_rate','segments','warnings','usage'}
    if not isinstance(value,dict) or set(value)!=required:raise AudioError('Invalid transcript fields')
    if value['schema_version']!=1 or value['status']!='complete' or value['sample_rate']!=16000:raise AudioError('Incomplete transcript')
    pin=manifest()
    if value['model']!={'id':pin['id'],'revision':pin['revision']}:raise AudioError('Unexpected speech model revision')
    duration=value['duration_seconds']
    if type(duration) not in (int,float) or not math.isfinite(duration) or not 0<duration<=1800:raise AudioError('Invalid audio duration')
    if not isinstance(value['segments'],list) or len(value['segments'])>10000:raise AudioError('Invalid segment count')
    prior=-1
    for s in value['segments']:
        if not isinstance(s,dict) or set(s)!={'start','end','speaker','text'}:raise AudioError('Invalid transcript segment')
        a,b=s['start'],s['end']
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (a,b)):raise AudioError('Nonfinite transcript time')
        if not 0<=a<=b<=duration+.1 or a<prior:raise AudioError('Invalid transcript time range/order')
        prior=a
        if not isinstance(s['speaker'],str) or not re.fullmatch(r'S\d{1,4}',s['speaker']):raise AudioError('Invalid anonymous speaker label')
        if not isinstance(s['text'],str) or not s['text'].strip():raise AudioError('Empty transcript segment')
    allowed={'missing_speaker_labels','empty_transcript_unverified','speaker_identity_unverified','utterance_timestamps_not_word_timestamps'}
    if not isinstance(value['warnings'],list) or any(not isinstance(w,str) or w not in allowed for w in value['warnings']):raise AudioError('Invalid warnings')
    usage=value['usage']
    if not isinstance(usage,dict) or set(usage)!={'prompt_tokens','generated_tokens','forward_calls','peak_allocated_bytes','inference_seconds'}:raise AudioError('Invalid usage fields')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in usage.values()):raise AudioError('Invalid usage values')
    if len(json.dumps(value,ensure_ascii=False).encode())>8*1024**2:raise AudioError('Transcript exceeds 8MiB')
    return value
