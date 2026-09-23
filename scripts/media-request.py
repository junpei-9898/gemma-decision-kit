"""Build a bounded inline request from a caller-selected local file."""
import argparse,base64,json
from pathlib import Path
from gemma_decision.core import validate
p=argparse.ArgumentParser();p.add_argument('type',choices=['image','video']);p.add_argument('file');p.add_argument('template');a=p.parse_args()
f=Path(a.file)
if f.stat().st_size>4*1024**2:p.error('Input file exceeds4MiB client limit')
mime={'.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.mp4':'video/mp4'}.get(f.suffix.lower())
if not mime or not mime.startswith(a.type+'/'):p.error('Unsupported extension')
b=json.loads(Path(a.template).read_text());b['media']={'type':a.type,'data':'data:'+mime+';base64,'+base64.b64encode(f.read_bytes()).decode()}
validate(b);print(json.dumps(b,ensure_ascii=False))
