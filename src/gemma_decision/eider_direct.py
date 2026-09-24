"""Direct pinned Eider semantics. Experimental; one bridge/backend per process.

No new server, no subprocess per request, no Python decision rules.
"""
# SPDX-License-Identifier: Apache-2.0
import ctypes
import json
import threading
import time
import secrets
from .context import MAX_REQUEST_TOKENS

_LOCK = threading.RLock()

class EiderBridge:
    def __init__(self, library, model_dir):
        self.lib = ctypes.CDLL(str(library))
        self.lib.eider_call.argtypes = [ctypes.c_char_p]
        self.lib.eider_call.restype = ctypes.c_void_p
        self.lib.eider_free.argtypes = [ctypes.c_void_p]
        self.lib.eider_free.restype = None
        self.call(op='init', model_dir=str(model_dir))

    def call(self, **command):
        with _LOCK:
            ptr = self.lib.eider_call(json.dumps(command, ensure_ascii=False, allow_nan=False).encode())
            if not ptr:raise RuntimeError('Null Eider response')
            try:result = json.loads(ctypes.string_at(ptr))
            finally:self.lib.eider_free(ptr)
            if 'error' in result:raise ValueError(result['error'])
            return result['ok']

class EiderDecision:
    def __init__(self, bridge, backend, context=8192):
        if not backend.decision_logits:raise ValueError('Requires raw selected logits backend')
        self.bridge, self.backend, self.context = bridge, backend, context

    def predict(self, body, *, token_budget=None):
        with _LOCK:
            start=time.perf_counter()
            from .eider_engine import validate
            validate(body)
            cap = MAX_REQUEST_TOKENS if token_budget is None else min(MAX_REQUEST_TOKENS, token_budget)
            if type(cap) is not int or cap < 1:raise ValueError('No remaining input token budget')
            media = body.get('media')
            request = {k:v for k,v in body.items() if k != 'media'}
            request.setdefault('model', 'local-gemma4')
            marker = None
            media_info = None
            if media is not None:
                from .media import inspect_media
                if not self.backend.media:raise ValueError('Restart with --media')
                media_info = inspect_media(media)
                marker = 'GDK_MEDIA_' + secrets.token_hex(16)
                while marker in request['state']:marker = 'GDK_MEDIA_' + secrets.token_hex(16)
                request['state'] += '\n' + marker + '\n'
            prepared=self.bridge.call(op='prepare', body=request)
            self.prepare_seconds=time.perf_counter()-start
            self.prepared=prepared
            try:
                payloads=[]; self.physical_input_tokens=0
                for branch in prepared['branches']:
                    tokens=prepared['prefix_tokens']+branch['suffix_tokens']
                    payload=tokens; counts=None
                    if media is not None:
                        payload,tokens,counts=self.backend.prepare_eider_media(media,tokens,marker,self.context-1,branch['suffix_tokens'])
                    if len(tokens)>=self.context:raise ValueError('Input exceeds context; no truncation applied')
                    self.physical_input_tokens+=len(tokens)
                    if self.physical_input_tokens>cap:raise ValueError('Aggregate input token budget exceeded; no inference performed')
                    payloads.append((branch,payload,counts))
                logits=[]
                for branch,payload,counts in payloads:
                    values=(self.backend.selected_media_logits(payload,branch['label_token_ids']) if media is not None
                            else self.backend.selected_logits(payload,branch['label_token_ids']))
                    logits.append({'question_id':branch['question_id'],'values':values})
                self.last_logits=logits
                result=self.bridge.call(op='finish', logits=logits)
                result['usage']['logical_input_tokens']=result['usage']['input_tokens'] if media is None else None
                result['usage']['input_tokens']=self.physical_input_tokens
                result['usage']['physical_input_tokens']=self.physical_input_tokens
                result['usage']['logical_tokens_include_media']=False
                result['probability_calibration']='uncalibrated'
                result['semantics']='eider-decision-v1'
                if media_info is not None:
                    result['media']={**media_info,'token_counts_by_question':{b['question_id']:c for b,_,c in payloads}}
                return result
            finally:self.bridge.call(op='discard')
