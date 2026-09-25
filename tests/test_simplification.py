# SPDX-License-Identifier: Apache-2.0
import io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from gemma_decision.cli import main
from gemma_decision.eider_engine import EiderEngine
from gemma_decision.inputs import analyze
from gemma_decision.inputs.aggregation import split_request
from test_contract import body

class SimplificationTests(unittest.TestCase):
    def test_removed_flags_fail_instead_of_silently_falling_back(self):
        for args in [['--semantics','legacy'],['--semantics','eider'],['--profile','speed'],['--hardware','gb10'],['--transcript-output','x'],['--audio','x.wav']]:
            with self.subTest(args=args),patch('sys.argv',['gemma-decision','predict']+args),patch('sys.stderr',io.StringIO()),patch('gemma_decision.cli.EiderEngine') as engine:
                with self.assertRaises(SystemExit) as error:main()
                self.assertEqual(error.exception.code,2);engine.assert_not_called()
    def test_bridge_is_required_without_legacy_fallback(self):
        with patch.dict('os.environ',{},clear=True),patch('gemma_decision.backends.load_backend') as load:
            with self.assertRaisesRegex(ValueError,'GEMMA_EIDER_LIBRARY'):EiderEngine('/model')
            load.assert_not_called()
    def test_unified_defaults_accept_all_eider_question_types(self):
        qs={'binary':{'type':'noul','instructions':'true?'},'rating':{'type':'score','instructions':'level?','criteria':['low','high']},'choice':{'type':'choice','instructions':'which?','criteria':{'a':'A','b':'B'}}}
        request={'state':'Context','questions':qs};split_request(request)
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source.txt';source.write_text('Full source')
            engine=Mock();engine.predict.return_value={'answers':{},'usage':{'input_tokens':3}}
            result=analyze(request,source,engine_factory=lambda media:engine)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(engine.predict.call_args.args[0]['questions'],qs)
        self.assertIn('Full source',engine.predict.call_args.args[0]['state'])
    def test_multiframe_typed_aggregation_rejected_before_load(self):
        request={'state':'x','questions':{'q':{'type':'noul','instructions':'true?'}}};factory=Mock()
        with patch('gemma_decision.inputs.pipeline.inspect_source',return_value=({'kind':'video','audio':True,'duration_seconds':12},None)),patch('gemma_decision.inputs.pipeline.digest',return_value='hash'),patch('gemma_decision.inputs.pipeline.obtain') as asr:
            with self.assertRaisesRegex(ValueError,'Multi-window'):analyze(request,'x',engine_factory=factory)
            factory.assert_not_called();asr.assert_not_called()
    def test_unified_cli_instantiates_eider_and_forwards_hardware(self):
        def workflow(request,source,**kwargs):
            kwargs['engine_factory'](media=True)
            return {'status':'complete','decision':{}}
        with patch('sys.argv',['gemma-decision','analyze','--source','x.mp4','--model-path','/model','--hardware','spark','--bridge-library','/lib']),patch('sys.stdin',io.StringIO(json.dumps(body()))),patch('sys.stdout',io.StringIO()),patch('gemma_decision.inputs.analyze',side_effect=workflow),patch('gemma_decision.cli.EiderEngine') as engine:
            main()
        engine.assert_called_once_with('/model',None,media=True,bridge_library='/lib',hardware='spark')
    def test_http_worker_configuration_has_no_removed_flags(self):
        with patch('sys.argv',['gemma-decision','serve-input','--model-path','/model','--bridge-library','/lib','--hardware','standard']),patch('gemma_decision.inputs.http.serve') as serve:
            main()
        options=serve.call_args.args[1]
        self.assertIn('standard',options);self.assertIn('/lib',options)
        self.assertNotIn('--semantics',options);self.assertNotIn('--profile',options)
