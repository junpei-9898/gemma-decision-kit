# SPDX-License-Identifier: Apache-2.0
"""Bounded inline media only. Never fetch URLs or open caller-supplied paths."""
import base64
import binascii
import io
import math
import tempfile


def validate_media(item):
    if not isinstance(item,dict) or set(item)!={'type','data'} or item['type'] not in ('image','video'):
        raise ValueError('media must contain type=image|video and an inline data URL')
    data=item['data']
    allowed={'image':('data:image/png;base64,','data:image/jpeg;base64,'),'video':('data:video/mp4;base64,')}
    if not isinstance(data,str) or len(data)>6*1024**2 or not data.startswith(allowed[item['type']]):
        raise ValueError('Unsupported media data URL or exceeds 6MiB encoded limit')
    try:raw=base64.b64decode(data.split(',',1)[1],validate=True)
    except (ValueError,binascii.Error) as exc:raise ValueError('Invalid base64 media') from exc
    if not raw:raise ValueError('Empty media')
    return raw


def inspect_media(item):
    raw=validate_media(item)
    if item['type']=='image':
        from PIL import Image, UnidentifiedImageError
        try:
            with Image.open(io.BytesIO(raw)) as im:
                width,height=im.size
                if im.format not in ('PNG','JPEG') or not 0<width<=1920 or not 0<height<=1920 or width*height>2073600:
                    raise ValueError('Image must be PNG/JPEG, at most 1920 per axis and 2073600 pixels')
                if getattr(im,'n_frames',1)!=1:raise ValueError('Animated images are not supported')
                im.verify()
        except (UnidentifiedImageError,OSError,SyntaxError) as exc:raise ValueError('Invalid image') from exc
        return {'type':'image','width':width,'height':height}
    import cv2
    # Private bounded temporary copy; no user-controlled pathname.
    with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
        f.write(raw);f.flush();cap=cv2.VideoCapture(f.name)
        try:
            fps=cap.get(cv2.CAP_PROP_FPS);frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width=cap.get(cv2.CAP_PROP_FRAME_WIDTH);height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if not cap.isOpened() or not all(math.isfinite(v) and v>0 for v in (fps,frames,width,height)):
                raise ValueError('Invalid MP4 video metadata')
            duration=frames/fps
            if duration>10.01 or frames>300 or width>1280 or height>720:
                raise ValueError('Video exceeds 10 seconds, 300 source frames or 1280x720')
            return {'type':'video','width':int(width),'height':int(height),'source_frames':int(frames),'duration_seconds':duration,'frame_selection':'pinned_vllm_default; sampled, not full-frame analysis','audio_processed':False}
        finally:cap.release()


def messages(item,text):
    kind=item['type']+'_url'
    return [{'role':'user','content':[{'type':kind,kind:{'url':item['data']}},{'type':'text','text':text}]}]
