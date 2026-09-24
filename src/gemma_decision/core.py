# SPDX-License-Identifier: Apache-2.0
import math
import threading
from .profiles import PROFILES
from .context import context_settings, MAX_REQUEST_TOKENS


def validate(body):
    if not isinstance(body, dict) or set(body) not in ({'state', 'questions'},{'state','questions','media'}):
        raise ValueError('Expected state, questions and optional media')
    if 'media' in body:
        from .media import validate_media
        validate_media(body['media'])
    max_chars = 200000 if 'media' in body else 2000000
    if not isinstance(body['state'], str) or not body['state'] or len(body['state']) > max_chars:
        raise ValueError(f'state must be nonempty text, at most {max_chars} characters')
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
    def __init__(self, profile, model_path, max_input_tokens=None, media=False):
        if not __debug__:
            raise RuntimeError("Python -O disables required runtime guards; use normal Python")
        limit, context, kv_bytes = context_settings(max_input_tokens, media)
        if profile not in PROFILES:
            raise ValueError('Unsupported profile or token limit')
        from .backends import load_backend
        self.backend = load_backend(profile, model_path, media=media, context=context, kv_bytes=kv_bytes)
        self.profile = profile
        self.limit = limit
        self.lock = threading.Lock()

    def predict(self, body, *, token_budget=None):
        validate(body)
        if token_budget is not None and (type(token_budget)!=int or token_budget<1):
            raise ValueError("No remaining input token budget")
        request_token_cap=min(MAX_REQUEST_TOKENS,token_budget) if token_budget is not None else MAX_REQUEST_TOKENS
        # All questions validated/tokenized before any inference; no truncation.
        with self.lock:
            prepared=[]
            total_tokens=0
            media_info=None
            if 'media' in body:
                if not getattr(self.backend,'media',False):raise ValueError('Restart with --media for image/video inputs')
                from .media import inspect_media, messages
                media_info=inspect_media(body['media'])
            for key,q in body['questions'].items():
                if media_info is not None:
                    payload,ids,counts=self.backend.prepare_media(messages(body['media'],prompt_text(body['state'],q)),self.limit)
                else:
                    ids=self.backend.tokenizer.apply_chat_template([{'role':'user','content':prompt_text(body['state'],q)}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
                    if not isinstance(ids,list):ids=ids['input_ids']
                    payload=ids;counts=None
                if len(ids)>self.limit:raise ValueError(f'Question {key} exceeds token limit; input was not truncated')
                total_tokens += len(ids)
                if total_tokens > request_token_cap:
                    raise ValueError(f'Request exceeds aggregate {request_token_cap} input tokens; no inference performed')
                prepared.append((key,q,ids,payload,counts))
            answers={}
            for key,q,ids,payload,counts in prepared:
                values=self.backend.score_media(payload) if media_info is not None else self.backend.score(ids)
                if len(values)!=3 or any(not math.isfinite(v) or v<0 for v in values) or abs(sum(values)-1)>1e-5:
                    raise RuntimeError('Invalid backend probability distribution')
                probabilities=dict(zip(q['criteria'],values))
                answers[key]={'type':'choice','choice':max(probabilities,key=probabilities.get),'probabilities':probabilities}
        result={'profile':self.profile,'model':PROFILES[self.profile]['model'],'answers':answers,'usage':{'input_tokens':sum(len(x[2]) for x in prepared)},'probability_calibration':'uncalibrated'}

        if media_info is not None:result['media']={**media_info,'token_counts_by_question':{x[0]:x[4] for x in prepared}}
        return result
