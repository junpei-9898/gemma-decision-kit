# SPDX-License-Identifier: Apache-2.0
# @unit evaluation
# @layer infrastructure
# @work-item-id WI-060
"""Research bridges only; does not change distributed Gemma API."""
import sys,json,math,copy,urllib.request
from pathlib import Path
W=Path('/work')
def criteria(q):
    if q['type']=='score':return {str(i):v for i,v in enumerate(q['criteria'])}
    return q['criteria']
def answer(q,p):
    result={'type':q['type'],'probabilities':p}
    if q['type']=='noul':result['noul']=p['true']
    elif q['type']=='score':result['score']=sum(int(k)*v for k,v in p.items())
    else:result['choice']=max(p,key=p.get)
    return result
class Gemma:
    def __init__(self):
        from gemma_decision.backends import SpeedBackend
        self.b=SpeedBackend('/gemma');self.n=3;self.forward_calls=None
    def predict(self,body):
        from gemma_decision.nv_head import install_candidates
        from gemma_decision.nv_runtime import TRIAL
        from vllm import SamplingParams
        out={};count=0
        for key,q in body['questions'].items():
            opts=criteria(q);letters='ABCDEFGH'[:len(opts)]
            if self.n!=len(opts):
                ids=[self.b.tokenizer.encode(c,add_special_tokens=False) for c in letters]
                assert all(len(x)==1 for x in ids);self.b.ids=[x[0] for x in ids]
                self.b.model.apply_model(lambda m:install_candidates(m,TRIAL,self.b.ids,None))
                self.b.params=SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=1,allowed_token_ids=self.b.ids,logprobs=len(opts),seed=0);self.n=len(opts)
            text=body['state']+'\n\n'+q['instructions']+'\n'+'\n'.join(c+'='+k+'：'+v for c,(k,v) in zip(letters,opts.items()))+'\n答えは'+'、'.join(letters)+'のいずれか1文字。'
            ids=self.b.tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            if not isinstance(ids,list):ids=ids['input_ids']
            assert all(type(i) is int for i in ids)
            assert len(ids)<=65535;count+=len(ids)
            p=self.b.score(ids);out[key]=answer(q,dict(zip(opts,p)))
        return {'answers':out,'usage':{'input_tokens':count}}
    def coverage(self,body):return {'truncated':False,'basis':'score verifies returned input token IDs; <=65535 before model'}
class Direct:
    def __init__(self,name):
        from decision_adapters import DecisionAdapter
        self.name=name;self.a=DecisionAdapter(name,W/'imported',json.loads((W/'imported/additional-weight-pins.json').read_text()))
    @property
    def forward_calls(self):return self.a.forward_calls
    def adapted(self,body):
        b=copy.deepcopy(body)
        for q in b['questions'].values():
            if self.name=='nanojev' and q['type']=='noul':q['type']='boolean'
            elif self.name=='semif':q['criteria']=criteria(q);q['type']='choice'
        return b
    def predict(self,body):
        b=self.adapted(body);r=self.a.predict(self.a.prepare(b))
        if self.name=='semif':r['answers']={k:answer(body['questions'][k],v['probabilities']) for k,v in r['answers'].items()}
        return r
    def coverage(self,body):
        b=self.adapted(body)
        if self.name!='laya':return self.a.coverage(b)
        from laya.common import render_options,build_sequence
        tok=self.a.tokenizer;engine=self.a.engine;out={}
        for k,q in b['questions'].items():
            internal=engine._to_internal(q);head=tok(q['type']+' question: '+q['instructions'],add_special_tokens=False)['input_ids'];opts=[tok(' '+o,add_special_tokens=False)['input_ids'] for o in render_options(internal)]
            clipped=[[tok.mask_token_id]+o[:48] for o in opts];limit=engine.cfg['head_max_len'];room=limit-sum(map(len,clipped))
            if room<16:
                per=max(4,(limit-16)//len(clipped));clipped=[o[:per] for o in clipped];room=limit-sum(map(len,clipped))
            kept=min(len(head),max(8,room));state=tok(b['state'],add_special_tokens=False)['input_ids'];capacity=max(0,engine.cfg['max_len']-(1+kept+1+sum(map(len,clipped))+1)-1)
            seq,_=build_sequence(tok,b['state'],internal,engine.cfg['max_len'],limit)
            out[k]={'input_tokens':len(seq),'truncated':kept<len(head) or len(state)>capacity or any(len(o)>len(c)-1 for o,c in zip(opts,clipped)),'head_full':len(head),'head_kept':kept,'state_full':len(state),'state_kept':min(len(state),capacity)}
        return out
class Http:
    forward_calls=None
    def __init__(self,name):self.name=name
    def predict(self,body):
        b=copy.deepcopy(body)
        if self.name=='qwen':b['model']='local-qwen3_5_moe';url='http://127.0.0.1:18080/v1/decisions'
        else:b.update(model='diffusiongemma',samples=1,steps=1,think=0);url='http://127.0.0.1:18091/v1/systemone'
        req=urllib.request.Request(url,json.dumps(b,ensure_ascii=False).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=120) as r:return json.load(r)
    def coverage(self,body):return {'truncated':False,'basis':'short fixture; configured8192; native server length rejection; prompt counts recorded from usage where exposed'}
