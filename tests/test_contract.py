import json, threading, unittest, urllib.request, urllib.error
from gemma_decision.eider_engine import validate
from gemma_decision.eider_direct import EiderDecision
from gemma_decision.cli import parse_json
from gemma_decision.server import make_server


def body():
    return {'state':'申請は期限内。','questions':{'q':{'type':'choice','instructions':'期限内？','criteria':{'yes':'期限内','no':'期限外','unknown':'不明'}}}}

class Bridge:
    """Protocol fake only: actual Rust semantics are tested by the native suite."""
    def __init__(self):self.ops=[];self.suffixes=None
    def call(self,**cmd):
        self.ops.append(cmd)
        if cmd['op']=='prepare':
            self.body=cmd['body'];qs=self.body['questions']
            return {'prefix_tokens':[1,2],'branches':[{'question_id':k,'suffix_tokens':self.suffixes[i] if self.suffixes else [3],'label_token_ids':[10,11,12]} for i,k in enumerate(qs)]}
        if cmd['op']=='finish':
            return {'usage':{'input_tokens':2+len(self.body['questions'])},'answers':{k:{'type':'choice','choice':next(iter(q['criteria'])),'probabilities':dict(zip(q['criteria'],[.7,.2,.1]))} for k,q in self.body['questions'].items()}}
        return {}

class Fake:
    decision_logits=True
    media=False
    hardware='spark'
    def __init__(self):self.calls=0
    def selected_logits(self,tokens,ids):self.calls+=1;return [2.,1.,0.]

def engine():return EiderDecision(Bridge(),Fake(),8193)

class ContractTests(unittest.TestCase):
    def test_result_is_constructed_by_bridge(self):
        e=engine();r=e.predict(body());self.assertEqual(r['answers']['q']['choice'],'yes');self.assertEqual(r['usage']['input_tokens'],3);self.assertEqual(e.backend.calls,1)
        self.assertEqual([x['op'] for x in e.bridge.ops],['prepare','finish','discard'])
        self.assertEqual(e.bridge.ops[1]['logits'],[{'question_id':'q','values':[2.,1.,0.]}])
    def test_unknown_fields_and_unsupported_types(self):
        b=body();b['questions']['q']['type']='boolean'
        with self.assertRaises(ValueError):validate(b)
        b=body();b['extra']=1
        with self.assertRaises(ValueError):validate(b)
    def test_choice_range_and_typed_contract(self):
        for n in [2,3,64]:
            b=body();b['questions']['q']['criteria']={str(i):str(i) for i in range(n)};validate(b)
        for n in [0,1,65]:
            b=body();b['questions']['q']['criteria']={str(i):str(i) for i in range(n)}
            with self.assertRaises(ValueError):validate(b)
        for q in [{'type':'noul','instructions':'true?'},{'type':'score','instructions':'level?','criteria':['low','high']}]:validate({'state':'x','questions':{'q':q}})
    def test_token_limit_before_inference(self):
        e=engine();e.context=3
        with self.assertRaises(ValueError):e.predict(body())
        self.assertEqual(e.backend.calls,0)
    def test_duplicate_json_rejected(self):
        with self.assertRaises(ValueError):parse_json('{"state":"a","state":"b"}')
    def test_http(self):
        e=engine();s=make_server(e,0);t=threading.Thread(target=s.serve_forever);t.start();base=f'http://127.0.0.1:{s.server_port}'
        try:
            self.assertEqual(json.load(urllib.request.urlopen(base+'/health'))['semantics'],'eider-decision-v1')
            req=urllib.request.Request(base+'/v1/decisions',json.dumps(body()).encode(),{'Content-Type':'application/json'})
            self.assertEqual(json.load(urllib.request.urlopen(req))['answers']['q']['choice'],'yes')
            req=urllib.request.Request(base+'/v1/decisions',b'{}',{'Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as cm:urllib.request.urlopen(req)
            self.assertEqual(cm.exception.code,400)
        finally:s.shutdown();t.join();s.server_close()

if __name__=='__main__':unittest.main()
