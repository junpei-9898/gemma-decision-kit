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
    p.add_argument('command',choices=['info','validate','predict','serve','transcribe','download-audio','analyze','serve-input'])
    p.add_argument('--semantics',choices=['eider','legacy'],default='legacy')
    p.add_argument('--bridge-library',help='Path to pinned Eider Rust bridge library')
    p.add_argument('--profile',choices=PROFILES,default='speed')
    p.add_argument('--model-path')
    p.add_argument('--media',action='store_true',help='Load vision encoder and validated image/video recipe')
    p.add_argument('--input',default='-',help='JSON path or stdin')
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--max-input-tokens',type=int,default=None,help='Text default 65535, maximum 262143; media default/maximum 8192; includes prompt framing')
    p.add_argument('--source',help='Local text/image/audio/video; automatic routing for analyze')
    p.add_argument('--cache-dir',help='Optional private (0700) transcript cache')
    p.add_argument('--max-decisions',type=int,default=512)
    p.add_argument('--max-total-tokens',type=int,default=1000000)
    p.add_argument('--audio',help='Local audio/video file; offline MOSS preprocessing')
    p.add_argument('--audio-model-path',help='Pinned local MOSS checkpoint')
    p.add_argument('--audio-python',help='Python executable in the isolated MOSS environment')
    p.add_argument('--audio-max-seconds',type=float,default=1800)
    p.add_argument('--audio-timeout',type=float,default=1800)
    p.add_argument('--audio-max-new-tokens',type=int,default=16384)
    p.add_argument('--transcript-output',help='New private transcript JSON file for predict --audio')
    p.add_argument('--output',help='New private JSON output file (predict/transcribe)')
    args=p.parse_args()
    if args.semantics=='eider':
        from .eider_engine import EiderEngine, validate as validator
        engine_factory=lambda media=False:EiderEngine(args.profile,args.model_path,args.max_input_tokens,media=media,bridge_library=args.bridge_library)
    else:
        validator=validate
        engine_factory=lambda media=False:DecisionEngine(args.profile,args.model_path,args.max_input_tokens,media=media)
    from .audio.contracts import AudioError
    from .audio.pipeline import transcribe,predict_audio,write_private_json
    def emit(value):
        if args.output:write_private_json(args.output,value)
        else:print(json.dumps(value,ensure_ascii=False))
    if args.audio and args.command not in ['transcribe','predict']:p.error('--audio supports transcribe/predict only; HTTP audio is not supported')
    if args.output and args.command not in ['transcribe','predict','analyze']:p.error('--output supports transcribe/predict/analyze only')
    if args.transcript_output and not (args.command=='predict' and args.audio):p.error('--transcript-output requires predict --audio')
    if args.output:
        from pathlib import Path
        if Path(args.output).exists():p.error('Output already exists')
    if args.source and args.command!='analyze':p.error('--source requires analyze')
    audio_options=dict(python=args.audio_python,max_seconds=args.audio_max_seconds,timeout=args.audio_timeout,max_new_tokens=args.audio_max_new_tokens)
    if args.command in ['transcribe','download-audio']:
        if not args.audio_model_path:p.error('--audio-model-path is required')
        if args.command=='transcribe' and not args.audio:p.error('--audio is required')
        try:
            if args.command=='download-audio':
                from .audio.model import download_model
                from contextlib import redirect_stdout
                with redirect_stdout(sys.stderr):value=download_model(args.audio_model_path)
                print(json.dumps(value));return
            emit(transcribe(args.audio,args.audio_model_path,**audio_options));return
        except AudioError as exc:p.error(str(exc))
        except Exception:p.error('Audio setup or runtime failed; input details suppressed')
    if args.command in ['analyze','serve-input']:
        if not args.model_path:p.error('--model-path is required')
        if args.audio or args.media or args.transcript_output:p.error('Unified input selects its own audio/media route')
        if args.command=='serve-input':
            from .inputs.http import serve
            options=['--semantics',args.semantics,'--model-path',args.model_path,'--profile',args.profile,
                     '--max-decisions',str(args.max_decisions),'--max-total-tokens',str(args.max_total_tokens)]
            for flag,value in [('--bridge-library',args.bridge_library),('--audio-model-path',args.audio_model_path),('--audio-python',args.audio_python),('--cache-dir',args.cache_dir)]:
                if value:options.extend([flag,value])
            if args.max_input_tokens is not None:options.extend(['--max-input-tokens',str(args.max_input_tokens)])
            serve(args.port,options);return
        if not args.source:p.error('--source is required')
        from .inputs import analyze
        from contextlib import redirect_stdout
        try:
            with redirect_stdout(sys.stderr):
                text=sys.stdin.read(8*1024**2+1) if args.input=='-' else open(args.input,encoding='utf-8').read(8*1024**2+1)
                if len(text.encode('utf-8'))>8*1024**2:raise ValueError()
                result=analyze(parse_json(text),args.source,
                    engine_factory=engine_factory,validator=validator,
                    audio_model_path=args.audio_model_path,audio_python=args.audio_python,cache_dir=args.cache_dir,
                    max_decisions=args.max_decisions,max_total_tokens=args.max_total_tokens)
            emit(result)
        except AudioError as exc:p.error(str(exc))
        except Exception as exc:p.error('Unified input failed ('+type(exc).__name__+'); private input details suppressed')
        if result['status']!='complete':raise SystemExit(2)
        return
    if args.command=='info':
        print(json.dumps(PROFILES,indent=2));return
    if args.command in ['validate','predict']:
        text=sys.stdin.read(8*1024**2+1) if args.input=='-' else open(args.input,encoding='utf-8').read(8*1024**2+1)
        if len(text.encode('utf-8'))>8*1024**2:p.error('Input exceeds 8MiB')
        try:body=validator(parse_json(text))
        except (ValueError,TypeError) as exc:p.error(str(exc))
        if args.command=='validate':print('{"valid":true}');return
    if not args.model_path:p.error('--model-path is required')
    if args.command=='predict' and args.audio:
        if not args.audio_model_path:p.error('--audio-model-path is required')
        from contextlib import redirect_stdout
        try:
            with redirect_stdout(sys.stderr):
                result=predict_audio(body,args.audio,args.audio_model_path,
                    engine_factory=lambda:engine_factory(media=args.media),validator=validator,
                    transcript_output=args.transcript_output,**audio_options)
            emit(result);return
        except AudioError as exc:p.error(str(exc))
        except Exception:p.error('Audio decision failed; input details suppressed')
    # Runtime libraries may print during initialization/inference; stdout is JSON only.
    from contextlib import redirect_stdout
    with redirect_stdout(sys.stderr):
        engine=engine_factory(media=args.media)
        if args.command=='predict':result=engine.predict(body)
    if args.command=='predict':emit(result);return
    from .server import serve
    serve(engine,args.port)

if __name__=='__main__':main()
