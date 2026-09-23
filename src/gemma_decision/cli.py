# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import sys
from .profiles import PROFILES
from .core import DecisionEngine, validate


def parse_json(text):
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError('Duplicate JSON key')
            out[key]=value
        return out
    return json.loads(text,object_pairs_hook=pairs)


def main():
    p=argparse.ArgumentParser(description='Local NVFP4 three-choice decisions')
    p.add_argument('command',choices=['info','validate','predict','serve'])
    p.add_argument('--profile',choices=PROFILES,default='speed')
    p.add_argument('--model-path')
    p.add_argument('--media',action='store_true',help='Load vision encoder and validated image/video recipe')
    p.add_argument('--input',default='-',help='JSON path or stdin')
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--max-input-tokens',type=int,default=8192)
    args=p.parse_args()
    if args.command=='info':
        print(json.dumps(PROFILES,indent=2));return
    if args.command in ['validate','predict']:
        text=sys.stdin.read(8*1024**2+1) if args.input=='-' else open(args.input,encoding='utf-8').read(8*1024**2+1)
        if len(text)>8*1024**2:p.error('Input exceeds 8MiB')
        try:body=validate(parse_json(text))
        except (ValueError,TypeError) as exc:p.error(str(exc))
        if args.command=='validate':print('{"valid":true}');return
    if not args.model_path:p.error('--model-path is required')
    # Runtime libraries may print during initialization/inference; stdout is JSON only.
    from contextlib import redirect_stdout
    with redirect_stdout(sys.stderr):
        engine=DecisionEngine(args.profile,args.model_path,args.max_input_tokens,media=args.media)
        if args.command=='predict':result=engine.predict(body)
    if args.command=='predict':print(json.dumps(result,ensure_ascii=False));return
    from .server import serve
    serve(engine,args.port)

if __name__=='__main__':main()
