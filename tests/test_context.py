import copy, unittest
from unittest.mock import patch
from gemma_decision.context import context_settings
from gemma_decision.eider_engine import EiderEngine, validate
from test_contract import engine, body

class ContextTests(unittest.TestCase):
    def test_native_tiers_and_output_reservation(self):
        for limit, context, gib in [(1,65536,3),(65535,65536,3),(65536,131072,4),(131071,131072,4),(131072,262144,8),(262143,262144,8)]:
            self.assertEqual(context_settings(limit),(limit,context,gib*1024**3))
        self.assertEqual(context_settings(),context_settings(65535))
        self.assertEqual(context_settings(media=True),(8192,16384,3*1024**3))
    def test_invalid_limits_before_loading(self):
        with patch('gemma_decision.backends.load_backend') as load, patch('gemma_decision.eider_engine.EiderBridge'), patch('gemma_decision.eider_engine.Path.is_file',return_value=True), patch('gemma_decision.backends._LOADED',False):
            for limit in [0,-1,262144,1000000,True,1.5]:
                with self.assertRaises(ValueError):EiderEngine('/model',limit)
            with self.assertRaises(ValueError):EiderEngine('/model',8193,media=True)
            load.assert_not_called()
    def test_selected_tier_reaches_backend(self):
        with patch('gemma_decision.backends.load_backend') as load, patch('gemma_decision.eider_engine.EiderBridge'), patch('gemma_decision.eider_engine.Path.is_file',return_value=True), patch('gemma_decision.backends._LOADED',False):
            e=EiderEngine('/model',131071,bridge_library='/lib')
            self.assertEqual(e.context,131072)
            load.assert_called_once_with('/model',media=False,context=131072,kv_bytes=4*1024**3,hardware="auto")
    def test_exact_limit_and_later_overflow(self):
        e=engine();e.context=4;e.predict(body());self.assertEqual(e.backend.calls,1)
        e.backend.calls=0;b=body();b['questions']['q2']=copy.deepcopy(b['questions']['q'])
        with patch.object(e.bridge,'suffixes',[[3],[3,4]]):
            with self.assertRaises(ValueError):e.predict(b)
        self.assertEqual(e.backend.calls,0)
    def test_aggregate_budget_rejects_before_any_inference(self):
        e=engine();b=body();b['questions']['q2']=copy.deepcopy(b['questions']['q'])
        with patch('gemma_decision.eider_direct.MAX_REQUEST_TOKENS',5):
            with self.assertRaisesRegex(ValueError,'Aggregate'):e.predict(b)
        self.assertEqual(e.backend.calls,0)
    def test_text_character_bound(self):
        b=body();b['state']='a'*2000000;validate(b)
        b['state']+='a'
        with self.assertRaises(ValueError):validate(b)
    def test_media_character_limit_unchanged(self):
        b=body();b['state']='a'*200001;b['media']={'type':'image','data':'data:image/png;base64,AA=='}
        with self.assertRaisesRegex(ValueError,'bounded nonempty'):validate(b)
    def test_cli_body_byte_limit(self):
        import io
        from gemma_decision.cli import main
        with patch('sys.argv',['gemma-decision','validate']),patch('sys.stdin',io.StringIO('あ'*2800000)),patch('sys.stderr',io.StringIO()):
            with self.assertRaises(SystemExit):main()
    def test_media_initialization_keeps_original_context_and_cache(self):
        with patch('gemma_decision.backends.load_backend') as load, patch('gemma_decision.eider_engine.EiderBridge'), patch('gemma_decision.eider_engine.Path.is_file',return_value=True), patch('gemma_decision.backends._LOADED',False):
            e=EiderEngine('/model',media=True,bridge_library='/lib')
            self.assertEqual(e.context,8193)
            load.assert_called_once_with('/model',media=True,context=16384,kv_bytes=3*1024**3,hardware="auto")
