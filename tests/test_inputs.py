# SPDX-License-Identifier: Apache-2.0
# @unit evaluation
# @layer infrastructure
# @work-item-id WI-063
import base64
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from gemma_decision.inputs.pipeline import analyze
from gemma_decision.inputs.sources import inspect_source,windows
from gemma_decision.inputs.http import make_server
from gemma_decision.audio.contracts import manifest,AudioError

BODY={'state':'Evaluate the supplied evidence.','questions':{'q':{'type':'choice','instructions':'Is the claim supported?', 'criteria':{'yes':'supported','no':'refuted','unknown':'insufficient'}}}}

class Engine:
    def __init__(self):self.bodies=[]
    def predict(self,body,*,token_budget):
        if token_budget<10:raise ValueError('budget')
        self.bodies.append(body)
        return {'answers':{'q':{'choice':'yes','probabilities':{'yes':1.,'no':0.,'unknown':0.}}},'usage':{'input_tokens':10}}


def transcript():
    pin=manifest()
    return {'schema_version':1,'status':'complete','model':{'id':pin['id'],'revision':pin['revision']},
            'duration_seconds':12.,'sample_rate':16000,'segments':[
                {'start':9.,'end':11.,'speaker':'S01','text':'boundary utterance'}],
            'warnings':['speaker_identity_unverified'],
            'usage':dict.fromkeys(['prompt_tokens','generated_tokens','forward_calls','peak_allocated_bytes','inference_seconds'],0)}

class InputsTests(unittest.TestCase):
    def test_speakerless_output_is_unknown_not_fabricated_identity(self):
        from gemma_decision.audio.worker import parse_with_warnings
        segments,warnings=parse_with_warnings('[0.00]hello[1.20]',2)
        self.assertEqual(segments[0]['speaker'],'S0000')
        self.assertEqual(warnings,['missing_speaker_labels'])
        for bad in ['prefix[0.00]hello[1.20]','[0.00]hello[1.20]tail','[0.00]hello[1.20][1.30][S01]mixed[1.90]']:
            with self.assertRaises(AudioError):parse_with_warnings(bad,2)

    def test_three_valued_aggregation(self):
        from gemma_decision.inputs.aggregation import reduce_windows,split_request
        def evidence(values):return [{'id':str(i),'decision':{'answers':{'q':{'choice':v}}}} for i,v in enumerate(values)]
        policy={'operator':'any','positive':'yes','negative':'no','unknown':'unknown'}
        for values,expected in [(['yes','unknown'],'yes'),(['no','no'],'no'),(['no','unknown'],'unknown')]:
            answer=reduce_windows(evidence(values),'q',policy)
            self.assertEqual(answer['choice'],expected);self.assertIsNone(answer['probabilities'])
        policy['operator']='all'
        for values,expected in [(['no','unknown'],'no'),(['yes','yes'],'yes'),(['yes','unknown'],'unknown')]:
            self.assertEqual(reduce_windows(evidence(values),'q',policy)['choice'],expected)
        split_request({**BODY,'aggregation':{'q':policy}})
        with self.assertRaises(ValueError):split_request({**BODY,'aggregation':{'q':{**policy,'unknown':'no'}}})

    def test_text_and_budget(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.txt';p.write_text('plain source');e=Engine()
            result=analyze(BODY,p,engine_factory=lambda media:e)
            self.assertEqual(result['status'],'complete');self.assertIn('plain source',e.bodies[0]['state'])
            result=analyze(BODY,p,engine_factory=lambda media:e,max_total_tokens=9)
            self.assertEqual(result['status'],'failed');self.assertIsNone(result['decision'])

    @unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg required')
    def test_real_av_windows_alignment_aggregation_and_partial(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'video.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=red:s=64x64:r=2:d=12',
                '-f','lavfi','-i','sine=frequency=440:duration=12','-c:v','libx264','-c:a','aac','-shortest',str(p)],check=True)
            e=Engine();events=[]
            def factory(media):events.append('gemma');self.assertTrue(media);return e
            def speech(*a,**kw):events.append('moss_finished');return transcript(),False
            with patch('gemma_decision.inputs.pipeline.obtain',side_effect=speech):
                result=analyze(BODY,p,engine_factory=factory,audio_model_path='unused')
                self.assertEqual(result['status'],'complete');self.assertEqual(events,['moss_finished','gemma'])
                self.assertEqual(len(e.bodies),3);self.assertEqual(len(result['evidence']),2)
                self.assertTrue(all(x['utterances'][0]['start']==9 for x in result['evidence']))
                self.assertEqual(result['coverage']['processed_windows'][-1]['end'],12)
                result=analyze(BODY,p,engine_factory=lambda media:Engine(),audio_model_path='unused',max_total_tokens=15)
                self.assertEqual(result['status'],'partial');self.assertIsNone(result['decision'])
                self.assertEqual(len(result['coverage']['unprocessed_windows']),1)
                policy={'q':{'operator':'any','positive':'yes','negative':'no','unknown':'unknown'}}
                logical=analyze({**BODY,'aggregation':policy},p,engine_factory=lambda media:Engine(),audio_model_path='unused',max_decisions=2)
                self.assertEqual(logical['usage']['completed_decisions'],2)
                self.assertEqual(logical['decision']['answers']['q']['choice'],'yes')
                self.assertIsNone(logical['decision']['answers']['q']['probabilities'])
                with self.assertRaises(AudioError):analyze(BODY,p,engine_factory=factory,max_decisions=2)
                with self.assertRaises(AudioError):analyze(BODY,p,engine_factory=factory)

    def test_aggregate_budget_exhaustion_keeps_no_final_decision(self):
        from gemma_decision.inputs.pipeline import _Run
        e=Engine();run=_Run(e,15)
        run.predict(BODY)
        with self.assertRaises(ValueError):run.predict(BODY)
        self.assertEqual(run.used,10);self.assertEqual(run.decisions,1)

    def test_image_auto_routing(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'renamed.bin';Image.new('RGB',(16,16),'red').save(p,format='PNG')
            flags=[];e=Engine()
            def factory(media):flags.append(media);return e
            result=analyze(BODY,p,engine_factory=factory)
            self.assertEqual(result['status'],'complete');self.assertEqual(flags,[True])
            self.assertEqual(e.bodies[0]['media']['type'],'image')

    def test_core_budget_preflight_prevents_any_forward(self):
        from test_contract import engine,body
        e=engine()
        with self.assertRaises(ValueError):e.predict(body(),token_budget=1)
        self.assertEqual(e.backend.calls,0)

    def test_windows_complete_coverage(self):
        spans=windows(1496.585)
        self.assertEqual(len(spans),150);self.assertEqual(spans[-1]['end'],1496.585)
        self.assertTrue(all(a['end']==b['start'] for a,b in zip(spans,spans[1:])))

    def test_cache_identity_and_private_permissions(self):
        from gemma_decision.inputs.transcripts import obtain
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'cache'
            with patch('gemma_decision.inputs.transcripts.transcribe',return_value=transcript()) as worker:
                first=obtain('source','model','digest',cache_dir=root)
                second=obtain('source','model','digest',cache_dir=root)
                self.assertFalse(first[1]);self.assertTrue(second[1]);self.assertEqual(worker.call_count,1)
                obtain('source','model','changed',cache_dir=root);self.assertEqual(worker.call_count,2)
            root.chmod(0o755)
            with self.assertRaises(AudioError):obtain('source','model','digest',cache_dir=root)

    def test_http_rejects_paths_and_uses_worker_result(self):
        def worker(args,*a,**kw):
            out=Path(args[args.index('--output')+1]);out.write_text(json.dumps({'status':'complete','decision':{'ok':True}}))
        with make_server(0,[]) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            url=f'http://127.0.0.1:{server.server_port}/v1/analyze'
            try:
                with patch('gemma_decision.inputs.http.run_owned',side_effect=worker) as run:
                    body={'request':BODY,'source':{'extension':'.txt','base64':base64.b64encode(b'test').decode()}}
                    req=Request(url,json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                    with urlopen(req) as r:self.assertEqual(json.load(r)['status'],'complete')
                    body['source']={'path':'/private/file'}
                    with self.assertRaises(HTTPError) as ctx:
                        urlopen(Request(url,json.dumps(body).encode(),headers={'Content-Type':'application/json'}))
                    self.assertEqual(ctx.exception.code,400);self.assertEqual(run.call_count,1)
            finally:server.shutdown();thread.join()

if __name__=='__main__':unittest.main()
