# SPDX-License-Identifier: Apache-2.0
"""Small loopback-only, serialized development API; not an Internet-facing gateway."""
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from .cli import parse_json


def make_server(engine, port):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup();self.connection.settimeout(30)
        def log_message(self,*args):pass  # No request/prompt logs.
        def reply(self,status,body):
            data=json.dumps(body,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path!='/health':self.reply(404,{'error':'not_found'});return
            self.reply(200,{'status':'ready','semantics':'eider-decision-v1'})
        def do_POST(self):
            if self.path!='/v1/decisions':self.reply(404,{'error':'not_found'});return
            try:
                if self.headers.get('Transfer-Encoding'):raise ValueError('Chunked requests are not supported')
                if self.headers.get_content_type()!='application/json':raise ValueError('Content-Type must be application/json')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=8*1024**2:raise ValueError('Body must be 1 byte to 8MiB')
                raw=self.rfile.read(length)
                if len(raw)!=length:raise ValueError('Incomplete body')
                body=parse_json(raw.decode('utf-8'))
                result=engine.predict(body)
            except (ValueError,TypeError,UnicodeError,TimeoutError) as exc:
                self.reply(400,{'error':'invalid_request','detail':str(exc)});return
            except Exception:
                self.reply(500,{'error':'inference_failed'});return
            self.reply(200,result)
    return HTTPServer(('127.0.0.1',port),Handler)


def serve(engine,port):
    with make_server(engine,port) as server:
        try:server.serve_forever()
        except KeyboardInterrupt:pass
