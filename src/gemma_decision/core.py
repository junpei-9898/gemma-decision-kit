# SPDX-License-Identifier: Apache-2.0
import math
import threading
from .profiles import PROFILES


def validate(body):
    if not isinstance(body, dict) or set(body) != {'state', 'questions'}:
        raise ValueError('Expected exactly state and questions')
    if not isinstance(body['state'], str) or not body['state'] or len(body['state']) > 200000:
        raise ValueError('state must be nonempty text, at most 200000 characters')
    questions = body['questions']
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError('questions must contain 1..64 named questions')
    for key, q in questions.items():
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ValueError('Invalid question id')
        if not isinstance(q, dict) or set(q) != {'type','instructions','criteria'} or q['type'] != 'choice':
            raise ValueError('Only choice questions with instructions and criteria are supported')
        if not isinstance(q['instructions'], str) or not q['instructions'] or len(q['instructions']) > 10000:
            raise ValueError('Invalid instructions')
        c = q['criteria']
        if not isinstance(c, dict) or len(c) != 3:
            raise ValueError('Exactly three ordered criteria are required')
        if any(not isinstance(k,str) or not k or len(k)>128 or not isinstance(v,str) or not v or len(v)>10000 for k,v in c.items()):
            raise ValueError('Criteria must map nonempty short names to text')
    return body


def prompt_text(state, q):
    return state+'\n\n'+q['instructions']+'\n'+'\n'.join(c+'='+name+'：'+desc for c,(name,desc) in zip('ABC',q['criteria'].items()))+'\n答えはA、B、Cのいずれか1文字。'


class DecisionEngine:
    def __init__(self, profile, model_path, max_input_tokens=8192):
        if not __debug__:
            raise RuntimeError("Python -O disables required runtime guards; use normal Python")
        if profile not in PROFILES or not 1 <= max_input_tokens <= 8192:
            raise ValueError('Unsupported profile or token limit')
        from .backends import load_backend
        self.backend = load_backend(profile, model_path)
        self.profile = profile
        self.limit = max_input_tokens
        self.lock = threading.Lock()

    def predict(self, body):
        validate(body)
        # All questions validated/tokenized before any inference; no truncation.
        with self.lock:
            prepared=[]
            for key,q in body['questions'].items():
                ids=self.backend.tokenizer.apply_chat_template([{'role':'user','content':prompt_text(body['state'],q)}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
                if not isinstance(ids,list):ids=ids['input_ids']
                if len(ids)>self.limit:raise ValueError(f'Question {key} exceeds token limit; input was not truncated')
                prepared.append((key,q,ids))
            answers={}
            for key,q,ids in prepared:
                values=self.backend.score(ids)
                if len(values)!=3 or any(not math.isfinite(v) or v<0 for v in values) or abs(sum(values)-1)>1e-5:
                    raise RuntimeError('Invalid backend probability distribution')
                probabilities=dict(zip(q['criteria'],values))
                answers[key]={'type':'choice','choice':max(probabilities,key=probabilities.get),'probabilities':probabilities}
        return {'profile':self.profile,'model':PROFILES[self.profile]['model'],'answers':answers,'usage':{'input_tokens':sum(len(x[2]) for x in prepared)},'probability_calibration':'uncalibrated'}
