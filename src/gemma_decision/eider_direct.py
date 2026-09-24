"""Direct pinned Eider semantics. Experimental; one bridge/backend per process.

No new server, no subprocess per request, no Python decision rules.
"""
# SPDX-License-Identifier: Apache-2.0
import ctypes
import json
import threading
import time

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

    def predict(self, body):
        with _LOCK:
            start=time.perf_counter()
            prepared=self.bridge.call(op='prepare', body=body)
            self.prepare_seconds=time.perf_counter()-start
            self.prepared=prepared
            try:
                if any(len(prepared['prefix_tokens'])+len(b['suffix_tokens'])>=self.context for b in prepared['branches']):
                    raise ValueError('Input exceeds context; no truncation applied')
                logits=[];self.physical_input_tokens=0
                for b in prepared['branches']:
                    tokens=prepared['prefix_tokens']+b['suffix_tokens']
                    values=self.backend.selected_logits(tokens,b['label_token_ids'])
                    logits.append({'question_id':b['question_id'],'values':values})
                    self.physical_input_tokens+=len(tokens)
                self.last_logits=logits
                return self.bridge.call(op='finish', logits=logits)
            finally:self.bridge.call(op='discard')
