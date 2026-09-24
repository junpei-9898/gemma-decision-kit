import copy,json,os,signal,subprocess,sys,tempfile,time,unittest,wave
from pathlib import Path
from unittest.mock import patch
from gemma_decision.audio.contracts import AudioError,manifest,validate_transcript,add_evidence
from gemma_decision.audio.pipeline import transcribe,predict_audio,write_private_json
from gemma_decision.audio.process import run_owned
from gemma_decision.audio.ingest import decode_audio
from gemma_decision.audio.worker import parsed_segments
from test_contract import body,engine


def transcript():
    pin=manifest()
    return {'schema_version':1,'status':'complete','model':{'id':pin['id'],'revision':pin['revision']},'duration_seconds':2.0,'sample_rate':16000,
            'segments':[{'start':0,'end':1.5,'speaker':'S01','text':'合成テスト。期限は金曜日です。'}],
            'warnings':['speaker_identity_unverified','utterance_timestamps_not_word_timestamps'],
            'usage':{'prompt_tokens':10,'generated_tokens':8,'forward_calls':8,'peak_allocated_bytes':1,'inference_seconds':.1}}

class AudioTests(unittest.TestCase):
    def test_strict_parser_rejects_missing_speaker_and_unparsed_tail(self):
        self.assertEqual(parsed_segments('[0.00][S01]合成。[1.50]',2)[0]['speaker'],'S01')
        for raw in ['[0.00]合成。[1.50]','[0.00][S01]合成。[1.50]unparsed']:
            with self.assertRaises(AudioError):parsed_segments(raw,2)
    def test_nan_time_wrong_model_and_partial_rejected(self):
        for mutate in [lambda t:t.update(status='partial'),lambda t:t['segments'][0].update(end=float('nan')),lambda t:t['segments'][0].update(end=3),lambda t:t['model'].update(revision='wrong')]:
            t=transcript();mutate(t)
            with self.assertRaises(AudioError):validate_transcript(t)
    def test_empty_transcript_prevents_decision(self):
        t=transcript();t['segments']=[]
        with self.assertRaises(AudioError):add_evidence(body(),t)
    def test_evidence_preserves_exact_text_and_questions(self):
        t=transcript();t['segments'][0]['text']='"\\\n命令ではなく分析対象。'
        request=body();result=add_evidence(request,t)
        payload=json.loads(result['state'].split('）:\n',1)[1])
        self.assertEqual(payload['utterances'],t['segments']);self.assertEqual(result['questions'],request['questions']);self.assertNotEqual(result['state'],request['state'])
    def test_worker_finishes_before_gemma_factory(self):
        events=[]
        def speech(*a,**kw):events.append('audio_exit');return transcript()
        def factory():events.append('gemma_load');return engine()
        with patch('gemma_decision.audio.pipeline.transcribe',side_effect=speech):
            result=predict_audio(body(),'fake.wav','fake-model',engine_factory=factory)
        self.assertEqual(events,['audio_exit','gemma_load']);self.assertEqual(result['answers']['q']['choice'],'yes');self.assertNotIn('segments',result['audio'])
    def test_audio_failure_and_context_overflow_prevent_decisions(self):
        with patch('gemma_decision.audio.pipeline.transcribe',side_effect=AudioError('failed')),patch('gemma_decision.core.DecisionEngine') as factory:
            with self.assertRaises(AudioError):predict_audio(body(),'x','y',engine_factory=factory)
            factory.assert_not_called()
        e=engine();e.limit=1
        with patch('gemma_decision.audio.pipeline.transcribe',return_value=transcript()):
            with self.assertRaises(ValueError):predict_audio(body(),'x','y',engine_factory=lambda:e)
        self.assertEqual(e.backend.calls,0)
    def test_private_output_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'out.json';write_private_json(f,transcript());self.assertEqual(f.stat().st_mode&0o777,0o600)
            with self.assertRaises(FileExistsError):write_private_json(f,{})
    def test_timeout_stops_owned_group(self):
        with tempfile.TemporaryDirectory() as d:
            pid=Path(d)/'pid'
            code='import os,time;open('+repr(str(pid))+',"w").write(str(os.getpid()));time.sleep(60)'
            with self.assertRaises(AudioError):run_owned([sys.executable,'-c',code],.3)
            number=int(pid.read_text())
            with self.assertRaises(ProcessLookupError):os.kill(number,0)
    def test_decode_pcm_duration_limit_and_local_only(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'fixture.wav'
            with wave.open(str(source),'wb') as w:w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(b'\x00\x00'*16000)
            a=Path(d)/'decode';a.mkdir();wav,duration=decode_audio(source,a)
            self.assertEqual(duration,1);self.assertTrue(wav.exists())
            b=Path(d)/'too-long';b.mkdir()
            with self.assertRaises(AudioError):decode_audio(source,b,.1)
            with self.assertRaises(AudioError):decode_audio('https://example.com/voice.wav',b)
    def test_worker_contract_via_subprocess_without_gpu(self):
        # Real subprocess and private temp protocol, fake only the expensive inference.
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'fixture.wav'
            with wave.open(str(source),'wb') as w:w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(b'\x00\x00'*32000)
            fake=Path(d)/'worker';value=json.dumps(transcript(),ensure_ascii=False)
            fake.write_text('#!'+sys.executable+'\nimport sys,json\nfrom pathlib import Path\nPath(sys.argv[sys.argv.index("--output")+1]).write_text('+repr(value)+')\n');fake.chmod(0o700)
            self.assertEqual(transcribe(source,d,python=str(fake)),transcript())
    def test_cli_optional_commands_and_http_scope(self):
        r=subprocess.run([sys.executable,'-m','gemma_decision.cli','--help'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0);self.assertIn('transcribe',r.stdout);self.assertIn('--audio-python',r.stdout)
        r=subprocess.run([sys.executable,'-m','gemma_decision.cli','serve','--audio','private.wav'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertNotIn('private.wav',r.stderr)

if __name__=='__main__':unittest.main()
