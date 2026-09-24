import base64,copy,json,threading,unittest
from pathlib import Path
from gemma_decision.core import DecisionEngine,validate
from gemma_decision.media import validate_media
from gemma_decision.profiles import PROFILES
from test_contract import body,engine

class MediaContractTests(unittest.TestCase):
    def test_nvfp4_only(self):self.assertEqual(list(PROFILES),['speed'])
    def test_reject_urls_paths_audio_extra_keys(self):
        for item in [{'type':'image','data':'https://example.com/a.png'}, {'type':'image','data':'file:///etc/passwd'}, {'type':'audio','data':'data:audio/wav;base64,AA=='}, {'type':'image','data':'data:image/png;base64,AA==','path':'x'}]:
            with self.assertRaises(ValueError):validate_media(item)
    def test_empty_corrupt_or_oversize_base64(self):
        for data in ['','?','A'* (6*1024**2)]:
            with self.assertRaises(ValueError):validate_media({'type':'image','data':'data:image/png;base64,'+data})
    def test_media_disabled_before_inference(self):
        e=engine();b=body();b['media']={'type':'image','data':'data:image/png;base64,AA=='}
        with self.assertRaisesRegex(ValueError,'--media'):e.predict(b)
        self.assertEqual(e.backend.calls,0)
    def test_public_examples_schema(self):
        root=Path(__file__).parents[1]
        for name in ['image','video']:validate(json.loads((root/f'examples/{name}-request.json').read_text()))
    def test_later_oversize_media_question_prevents_all_inference(self):
        from unittest.mock import patch
        e=engine();e.backend.media=True
        e.backend.prepare_media=lambda msg,limit:({},[1,2,3],{'image':3})
        e.backend.score_media=lambda payload: self.fail('Must reject before inference')
        e.limit=2;b=body();b['media']={'type':'image','data':'data:image/png;base64,AA=='}
        with patch('gemma_decision.media.inspect_media',return_value={'type':'image'}):
            with self.assertRaises(ValueError):e.predict(b)
    def test_prepares_all_questions_before_gpu(self):
        from unittest.mock import patch
        e=engine();e.backend.media=True;prepared=[]
        def prepare(msg,limit):
            prepared.append(msg)
            if len(prepared)==2:raise ValueError('second too long')
            return {},[1],{'image':1}
        e.backend.prepare_media=prepare;e.backend.score_media=lambda payload:self.fail('partial inference')
        b=body();b['questions']['q2']=copy.deepcopy(b['questions']['q']);b['media']={'type':'image','data':'data:image/png;base64,AA=='}
        with patch('gemma_decision.media.inspect_media',return_value={'type':'image'}):
            with self.assertRaises(ValueError):e.predict(b)
        self.assertEqual(len(prepared),2)
    def test_cli_forwards_media_flag(self):
        from unittest.mock import patch,MagicMock
        from gemma_decision.cli import main
        request=json.dumps(body());fake=MagicMock();fake.predict.return_value={'ok':True}
        with patch('sys.argv',['gemma-decision','predict','--media','--model-path','/model']),patch('sys.stdin',__import__('io').StringIO(request)),patch('sys.stdout',__import__('io').StringIO()),patch('gemma_decision.cli.DecisionEngine',return_value=fake) as init:
            main()
        init.assert_called_once_with('speed','/model',None,media=True)
