# SPDX-License-Identifier: Apache-2.0
# @unit evaluation
# @layer infrastructure
# @work-item-id WI-017
"""Thin schema bridges to audited, pinned upstream implementations; no gold access."""
import json
import sys
from pathlib import Path


def semif_rows(request):
    return [dict(id=qid,state=request['state'],question=q['instructions'],
                 options=[dict(id=k,description=v) for k,v in q['criteria'].items()])
            for qid,q in request['questions'].items()]


def nano_payload(request):
    return {'states':[dict(id='input',state=request['state'],questions=request['questions'])]}


class DecisionAdapter:
    def __init__(self, name, root, pins):
        self.name=name;self.root=Path(root);self.forward_calls=0
        if name=='semif':
            sys.path.insert(0,str(self.root/'sources/semif/src'))
            from semif_phase1.core import load_causal_model
            self.model,self.tokenizer,self.metadata=load_causal_model(str(self.root/'models/semif'),pins[name]['revision'])
        elif name=='laya':
            sys.path.insert(0,str(self.root/'sources/laya'))
            from laya import load
            self.engine=load(str(self.root/'models/laya-runtime'),device='cuda:0')
            self.model=self.engine.model;self.tokenizer=self.engine.tok
            self.metadata={'config':self.engine.cfg,'device':str(self.engine.device),'amp_dtype':str(self.engine.dtype)}
        elif name=='nanojev':
            sys.path.insert(0,str(self.root/'sources/nanojev/scripts'))
            from predict_toy_decisions import DecisionPredictor
            self.engine=DecisionPredictor(str(self.root/'models/nanojev'),max_length=8192,precision='bf16')
            self.model=self.engine.model;self.tokenizer=self.engine.tokenizer
            self.metadata={'run_config':self.engine.run_config,'evaluation_max_length':8192,'parameter_storage':'float32','forward_autocast':'bfloat16'}
        else:raise ValueError(name)
        self.model.register_forward_pre_hook(self._count)

    def _count(self,*args):
        self.forward_calls+=1

    def prepare(self,request):
        if self.name=='semif':return semif_rows(request)
        if self.name=='nanojev':return nano_payload(request)
        return {'state':request['state'],'questions':request['questions']}

    def predict(self,payload,shared=False):
        if self.name=='semif':
            from semif_phase1.direct import score
            if shared:
                from semif_phase1.shared import score_shared
                results,timing=score_shared(self.model,self.tokenizer,payload,self.metadata,8192)
            else:
                results=[score(self.model,self.tokenizer,row,self.metadata,8192) for row in payload];timing=None
            answers={row['id']:{'type':'choice','choice':row['option_ids'][max(range(len(row['probabilities'])),key=row['probabilities'].__getitem__)],'probabilities':dict(zip(row['option_ids'],row['probabilities']))} for row in results}
            return {'answers':answers,'upstream':results,'shared_timing':timing,'usage':{'input_tokens':sum(x['input_tokens'] for x in results),'output_tokens':0}}
        if self.name=='nanojev':
            result=self.engine.predict(payload)
            return {'answers':result['states'][0]['answers'],'upstream':result}
        result=self.engine.predict(**payload)
        if str(self.engine.device)!='cuda:0':raise RuntimeError('Unexpected Laya CPU fallback; stop GPU comparison')
        return result

    def coverage(self,request):
        if self.name=='laya':
            from laya.common import serialize_state,render_options,build_sequence
            tok=self.tokenizer;out={};limit=self.engine.cfg.get('max_len',512);head_limit=self.engine.cfg.get('head_max_len',192)
            for qid,q in request['questions'].items():
                internal=self.engine._to_internal(q);mask=tok.mask_token
                head=tok('choice question: '+q['instructions'].replace(mask,' '),add_special_tokens=False)['input_ids']
                original_opts=[tok(' '+o.replace(mask,' '),add_special_tokens=False)['input_ids'] for o in render_options(internal)]
                opts=[[tok.mask_token_id]+o[:48] for o in original_opts];budget=head_limit-sum(map(len,opts))
                if budget<16:
                    per=max(4,(head_limit-16)//len(opts));opts=[o[:per] for o in opts];budget=head_limit-sum(map(len,opts))
                kept_head=min(len(head),max(8,budget));room=max(0,limit-(1+kept_head+1+sum(map(len,opts))+1)-1)
                state=tok(serialize_state(request['state']).replace(mask,' '),add_special_tokens=False)['input_ids']
                seq,_=build_sequence(tok,request['state'],internal,limit,head_limit)
                out[qid]={'head_tokens_full':len(head),'head_tokens_kept':kept_head,'state_tokens_full':len(state),'state_tokens_kept':min(len(state),room),'option_tokens_full':[len(x) for x in original_opts],'option_tokens_kept':[len(x)-1 for x in opts],'input_tokens':len(seq),'truncated':kept_head<len(head) or len(state)>room or any(len(a)>len(b)-1 for a,b in zip(original_opts,opts))}
            return out
        if self.name=='nanojev':
            from predict_toy_decisions import prepare_examples
            examples=prepare_examples(nano_payload(request),self.tokenizer,self.engine.limit)
            return {x['qid']:{'input_tokens':sum(map(len,x['leaf_tokens'])),'max_path_tokens':max(map(len,x['leaf_tokens'])),'truncated':False} for x in examples}
        from semif_phase1.direct import encode_prompt
        return {x['id']:{'input_tokens':len(encode_prompt(self.tokenizer,x,8192)[0]),'truncated':False} for x in semif_rows(request)}
