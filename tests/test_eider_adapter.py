# SPDX-License-Identifier: Apache-2.0
import copy,json,unittest
from unittest.mock import Mock,patch
from gemma_decision.eider_engine import validate
from gemma_decision.eider_direct import EiderDecision
from gemma_decision.inputs.aggregation import split_request

BODY={'state':'Evidence','questions':{'q':{'type':'noul','instructions':'True?'}}}
class Bridge:
    def __init__(self,branches=1):self.ops=[];self.branches=branches
    def call(self,**command):
        self.ops.append(command)
        if command['op']=='prepare':return {'prefix_tokens':[1,2], 'branches':[{'question_id':str(i),'suffix_tokens':[3], 'label_token_ids':[10,11]} for i in range(self.branches)]}
        if command['op']=='finish':return {'usage':{'input_tokens':2+self.branches},'answers':{str(i):{'type':'noul','noul':.5} for i in range(self.branches)}}
        return {}
class EiderTests(unittest.TestCase):
    def test_optimized_python_refused_before_load(self):
        import subprocess,sys
        r=subprocess.run([sys.executable,'-O','-c',"from gemma_decision.eider_engine import EiderEngine; EiderEngine('/missing')"],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn('runtime guards',r.stderr)
    def backend(self):return Mock(decision_logits=True,media=False,selected_logits=Mock(return_value=[0.,0.]))
    def test_all_branches_budget_preflight_and_discard(self):
        b=Bridge(2);gpu=self.backend();e=EiderDecision(b,gpu)
        with self.assertRaises(ValueError):e.predict(BODY,token_budget=5)
        gpu.selected_logits.assert_not_called();self.assertEqual(b.ops[-1]['op'],'discard')
        r=e.predict(BODY,token_budget=6)
        self.assertEqual(r['usage']['input_tokens'],6);self.assertEqual(r['usage']['logical_input_tokens'],4)
        self.assertEqual(b.ops[-1]['op'],'discard')
    def test_failure_discards_pending_state(self):
        b=Bridge();gpu=self.backend();gpu.selected_logits.side_effect=RuntimeError('GPU failure')
        with self.assertRaises(RuntimeError):EiderDecision(b,gpu).predict(BODY)
        self.assertEqual(b.ops[-1]['op'],'discard')
    def test_bad_envelope_before_bridge(self):
        b=Bridge();e=EiderDecision(b,self.backend())
        for body in [{**BODY,'model':'another-model'},{**BODY,'extra':1},{**BODY,'state':''},{**BODY,'questions':{'q':{'type':'noul','instructions':'x','criteria':None}}}]:
            with self.assertRaises(ValueError):e.predict(body)
        self.assertFalse(b.ops)
    def test_media_all_branches_preflight(self):
        b=Bridge(2);gpu=self.backend();gpu.media=True
        gpu.prepare_eider_media.side_effect=[({'prompt_token_ids':[1]*4},[1]*4,{'image':2}),ValueError('too long')]
        body={**BODY,'media':{'type':'image','data':'data:image/png;base64,AA=='}}
        with patch('gemma_decision.media.inspect_media',return_value={'type':'image'}):
            with self.assertRaises(ValueError):EiderDecision(b,gpu).predict(body)
        gpu.selected_media_logits.assert_not_called();gpu.selected_logits.assert_not_called()
        self.assertEqual(b.ops[-1]['op'],'discard')
        self.assertNotIn('GDK_MEDIA_',body['state'])
    def test_typed_criteria_and_aggregation(self):
        for q in [{'type':'choice','instructions':'x','criteria':{'a':None,'b':'B'}},{'type':'score','instructions':{'rubric':'x'},'criteria':['none','some']},BODY['questions']['q']]:
            validate({**BODY,'questions':{'q':q}})
        with self.assertRaises(ValueError):split_request({**BODY,'aggregation':{'q':{'operator':'any','positive':'true','negative':'false','unknown':'unknown'}}},validate)
    def test_cli_eider_media_explicit(self):
        from gemma_decision.cli import main
        import io
        with patch('sys.argv',['gemma-decision','predict','--media','--model-path','/model','--bridge-library','/lib']),patch('sys.stdin',io.StringIO(json.dumps(BODY))),patch('sys.stdout',io.StringIO()),patch('gemma_decision.cli.EiderEngine') as init:
            init.return_value.predict.return_value={}
            main()
        init.assert_called_once_with('/model',None,media=True,bridge_library='/lib',hardware='auto')
if __name__=='__main__':unittest.main()
