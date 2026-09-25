# SPDX-License-Identifier: Apache-2.0
# @unit evaluation
# @layer infrastructure
# @work-item-id WI-063
"""Serialized loopback inline-input API; one isolated inference process per request."""
import base64
import binascii
import json
import os
from pathlib import Path
import sys
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from ..audio.pipeline import write_private_json
from ..audio.process import run_owned
from ..cli import parse_json
from .aggregation import split_request


def make_server(port, worker_options):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup(); self.connection.settimeout(30)
        def log_message(self,*args): pass
        def reply(self,status,value):
            data=json.dumps(value,ensure_ascii=False).encode()
            self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            self.reply(200,{'status':'ready','model_loading':'per_request'}) if self.path=='/health' else self.reply(404,{'error':'not_found'})
        def do_POST(self):
            if self.path!='/v1/analyze': self.reply(404,{'error':'not_found'}); return
            try:
                if self.headers.get('Transfer-Encoding') or self.headers.get_content_type()!='application/json': raise ValueError()
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=8*1024**2: raise ValueError()
                raw=self.rfile.read(length)
                if len(raw)!=length: raise ValueError()
                body=parse_json(raw.decode('utf-8'))
                if not isinstance(body,dict) or set(body)!={'request','source'}: raise ValueError()
                split_request(body['request'])
                source=body['source']
                if not isinstance(source,dict) or set(source)!={'extension','base64'}: raise ValueError()
                if source['extension'] not in {'.txt','.md','.png','.jpg','.jpeg','.wav','.flac','.mp3','.m4a','.aac','.ogg','.opus','.mp4','.mov','.mkv','.webm'}: raise ValueError()
                if not isinstance(source['base64'],str) or len(source['base64'])>6*1024**2: raise ValueError()
                data=base64.b64decode(source['base64'],validate=True)
                if not data: raise ValueError()
            except (ValueError,TypeError,KeyError,UnicodeError,TimeoutError,binascii.Error):
                self.reply(400,{'error':'invalid_input'}); return
            try:
                with tempfile.TemporaryDirectory(prefix='gemma-http-input-') as d:
                    root=Path(d); os.chmod(root,0o700)
                    path=root/('source'+source['extension'])
                    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                    with os.fdopen(fd,'wb') as f:f.write(data)
                    write_private_json(root/'request.json',body['request'])
                    args=[sys.executable,'-m','gemma_decision.cli','analyze',
                          '--source',str(path),'--input',str(root/'request.json'),'--output',str(root/'result.json')]+worker_options
                    # CLI uses exit2 on partial; retain its structured failure, never accept missing output.
                    try:run_owned(args,3600)
                    except Exception:
                        if not (root/'result.json').is_file(): raise
                    result_path=root/'result.json'
                    if result_path.stat().st_size>32*1024**2:raise ValueError()
                    result=json.loads(result_path.read_text())
                self.reply(200 if result['status']=='complete' else 422,result)
            except Exception:self.reply(500,{'error':'processing_failed'})
    return HTTPServer(('127.0.0.1',port),Handler)


def serve(port,options):
    with make_server(port,options) as server:
        try: server.serve_forever()
        except KeyboardInterrupt: pass
