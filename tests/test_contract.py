import json, threading, unittest, urllib.request, urllib.error
from gemma_decision.core import DecisionEngine, validate, prompt_text
from gemma_decision.cli import parse_json
from gemma_decision.server import make_server


def body():
    return {'state':'申請は期限内。','questions':{'q':{'type':'choice','instructions':'期限内？','criteria':{'yes':'期限内','no':'期限外','unknown':'不明'}}}}

class Fake:
    def __init__(self):self.tokenizer=self;self.calls=0
    def apply_chat_template(self,*args,**kwargs):return [1,2,3]
    def score(self,tokens):self.calls+=1;return [.7,.2,.1]

def engine():
    e=DecisionEngine.__new__(DecisionEngine);e.backend=Fake();e.limit=8192;e.lock=threading.Lock();e.profile='memory';return e

class ContractTests(unittest.TestCase):
    def test_python_optimized_mode_rejected_before_gpu_load(self):
        import subprocess,sys
        r=subprocess.run([sys.executable,"-O","-c","from gemma_decision.core import DecisionEngine; DecisionEngine('memory','/nonexistent')"],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn("runtime guards",r.stderr)
    def test_order_mapping_and_usage(self):
        e=engine();r=e.predict(body());self.assertEqual(r['answers']['q']['choice'],'yes');self.assertEqual(r['usage']['input_tokens'],3);self.assertEqual(e.backend.calls,1)
    def test_unknown_fields_and_unsupported_types(self):
        b=body();b['questions']['q']['type']='boolean'
        with self.assertRaises(ValueError):validate(b)
        b=body();b['extra']=1
        with self.assertRaises(ValueError):validate(b)
    def test_three_choices_required(self):
        b=body();b['questions']['q']['criteria'].pop('unknown')
        with self.assertRaises(ValueError):validate(b)
    def test_token_limit_before_inference(self):
        e=engine();e.limit=2
        with self.assertRaises(ValueError):e.predict(body())
        self.assertEqual(e.backend.calls,0)
    def test_bad_probability_rejected(self):
        e=engine();e.backend.score=lambda x:[float('nan'),.2,.1]
        with self.assertRaises(RuntimeError):e.predict(body())
    def test_duplicate_json_rejected(self):
        with self.assertRaises(ValueError):parse_json('{"state":"a","state":"b"}')
    def test_frozen_prompt_shape(self):
        b=body();self.assertEqual(prompt_text(b['state'],b['questions']['q']),'申請は期限内。\n\n期限内？\nA=yes：期限内\nB=no：期限外\nC=unknown：不明\n答えはA、B、Cのいずれか1文字。')
    def test_http(self):
        e=engine();s=make_server(e,0);t=threading.Thread(target=s.serve_forever);t.start();base=f'http://127.0.0.1:{s.server_port}'
        try:
            self.assertEqual(json.load(urllib.request.urlopen(base+'/health'))['profile'],'memory')
            req=urllib.request.Request(base+'/v1/decisions',json.dumps(body()).encode(),{'Content-Type':'application/json'})
            self.assertEqual(json.load(urllib.request.urlopen(req))['answers']['q']['choice'],'yes')
            req=urllib.request.Request(base+'/v1/decisions',b'{}',{'Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as cm:urllib.request.urlopen(req)
            self.assertEqual(cm.exception.code,400)
        finally:s.shutdown();t.join();s.server_close()

if __name__=='__main__':unittest.main()
