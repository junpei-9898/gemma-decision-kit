# @unit evaluation
# @layer infrastructure
# @work-item-id WI-059
# SPDX-License-Identifier: Apache-2.0
import json
import math
from pathlib import Path
import wave
from .contracts import AudioError
from .process import run_owned

def decode_audio(source,directory,max_seconds=1800):
    if type(max_seconds) not in (int,float) or not math.isfinite(max_seconds) or not 0<max_seconds<=1800:raise AudioError('Audio limit must be in (0, 1800] seconds')
    path=Path(source).expanduser().resolve()
    if not path.is_file() or not 0<path.stat().st_size<=4*1024**3:raise AudioError('Input must be a local nonempty file at most 4GiB')
    if path.suffix.lower() not in {'.wav','.flac','.mp3','.m4a','.aac','.ogg','.opus','.mp4','.mov','.mkv','.webm'}:raise AudioError('Unsupported audio/video format')
    directory=Path(directory);probe=directory/'probe.json'
    with probe.open('wb') as out:
        run_owned(['ffprobe','-v','error','-protocol_whitelist','file,pipe','-select_streams','a:0','-show_entries','stream=codec_type,start_time:format=duration,start_time','-of','json',str(path)],30,stdout=out)
    try:
        if probe.stat().st_size>65536:raise ValueError()
        info=json.loads(probe.read_text())
        if not info.get('streams'):raise ValueError()
        a=float(info['streams'][0].get('start_time',0));b=float(info.get('format',{}).get('start_time',0))
        if not all(math.isfinite(v) for v in (a,b)):raise ValueError()
        if abs(a-b)>.1:raise AudioError('Audio/video start offset exceeds supported 100ms')
        duration=info.get('format',{}).get('duration')
        if duration is not None and (not math.isfinite(float(duration)) or float(duration)>max_seconds+.1):raise AudioError('Audio exceeds duration limit; not truncated')
    except AudioError:raise
    except (ValueError,KeyError,TypeError):raise AudioError('Invalid audio metadata') from None
    wav=directory/'audio.wav'
    run_owned(['ffmpeg','-nostdin','-v','error','-protocol_whitelist','file,pipe','-i',str(path),'-map','0:a:0','-vn','-map_metadata','-1','-ac','1','-ar','16000','-t',str(max_seconds+1),'-c:a','pcm_s16le','-n',str(wav)],180)
    with wave.open(str(wav)) as audio:
        duration=audio.getnframes()/audio.getframerate()
        if audio.getnchannels()!=1 or audio.getframerate()!=16000 or not 0<duration<=max_seconds:raise AudioError('Decoded audio exceeds bounds; not truncated')
    wav.chmod(0o600)
    return wav,duration
