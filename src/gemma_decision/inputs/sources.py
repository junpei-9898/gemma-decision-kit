# SPDX-License-Identifier: Apache-2.0
# @unit evaluation
# @layer infrastructure
# @work-item-id WI-063
"""Inspect actual local content and build bounded, time-indexed visual inputs."""
import base64
import hashlib
import json
import math
from pathlib import Path
from ..audio.contracts import AudioError
from ..audio.process import run_owned
from ..media import inspect_media


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def inspect_source(source, directory):
    path = Path(source).expanduser().resolve()
    if not path.is_file() or not 0 < path.stat().st_size <= 4*1024**3:
        raise AudioError('Source must be a local nonempty file at most 4GiB')
    with path.open('rb') as f: head = f.read(16)
    if head.startswith(b'\x89PNG\r\n\x1a\n') or head.startswith(b'\xff\xd8\xff'):
        mime = 'png' if head.startswith(b'\x89PNG') else 'jpeg'
        item = inline(path, 'image', mime)
        inspect_media(item)
        return {'kind':'image', 'audio':False, 'duration_seconds':None, 'origin_seconds':0}, item
    # Plain UTF-8 text has an explicit suffix; binary media use actual stream inspection.
    if path.suffix.lower() in {'.txt', '.md'}:
        if path.stat().st_size > 8*1024**2: raise AudioError('Text file exceeds 8MiB')
        try: content = path.read_text(encoding='utf-8-sig')
        except UnicodeError: raise AudioError('Text input must be UTF-8') from None
        if not content.strip() or '\x00' in content: raise AudioError('Invalid text input')
        return {'kind':'text', 'audio':False, 'duration_seconds':None, 'origin_seconds':0}, content
    probe = Path(directory)/'source-probe.json'
    with probe.open('wb') as out:
        run_owned(['ffprobe','-v','error','-protocol_whitelist','file,pipe','-show_entries',
                   'format=format_name,duration,start_time:stream=codec_type,start_time:stream_disposition=attached_pic',
                   '-of','json',str(path)],30,stdout=out)
    if probe.stat().st_size > 65536: raise AudioError('Source metadata exceeds limit')
    try:
        info = json.loads(probe.read_text()); fmt = info['format']
        allowed = {'mov','mp4','m4a','3gp','3g2','mj2','matroska','webm','wav','flac','mp3','ogg','aac'}
        if not set(fmt['format_name'].split(',')) <= allowed: raise ValueError()
        video = [s for s in info['streams'] if s['codec_type']=='video' and not s.get('disposition',{}).get('attached_pic')]
        audio = [s for s in info['streams'] if s['codec_type']=='audio']
        if len(video)>1 or len(audio)>1 or not (video or audio): raise ValueError()
        duration = float(fmt['duration']); origin = float(fmt.get('start_time',0))
        if not math.isfinite(duration) or not 0<duration<=1800 or not math.isfinite(origin): raise ValueError()
        offsets = [float(s.get('start_time',origin))-origin for s in video+audio]
        if any(not math.isfinite(t) or abs(t)>.1 for t in offsets):
            raise AudioError('Track start offsets exceed supported 100ms; no implicit synchronization')
    except (ValueError,KeyError,TypeError): raise AudioError('Unsupported container, tracks or duration') from None
    return {'kind':'video' if video else 'audio','audio':bool(audio),
            'duration_seconds':duration,'origin_seconds':origin}, None


def inline(path, kind, subtype):
    if Path(path).stat().st_size > (6*1024**2-64)*3//4: raise AudioError('Prepared media exceeds inline limit')
    return {'type':kind,'data':f'data:{kind}/{subtype};base64,'+base64.b64encode(Path(path).read_bytes()).decode()}


def windows(duration):
    return [{'id':f'w{i:04d}', 'start':float(i*10), 'end':min(duration,(i+1)*10.)}
            for i in range(math.ceil(duration/10))]


def clip(source, window, directory):
    target = Path(directory)/(window['id']+'.mp4')
    run_owned(['ffmpeg','-nostdin','-v','error','-protocol_whitelist','file,pipe',
               '-ss',str(window['start']),'-i',str(Path(source).resolve()),'-t',str(window['end']-window['start']),
               '-map','0:v:0','-an','-map_metadata','-1','-vf',
               'scale=640:360:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1,fps=2',
               '-c:v','libx264','-preset','ultrafast','-crf','23','-pix_fmt','yuv420p','-threads','2','-n',str(target)],180)
    item = inline(target,'video','mp4'); inspect_media(item)
    return item
