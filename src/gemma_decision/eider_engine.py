# SPDX-License-Identifier: Apache-2.0
"""Distribution adapter; Eider Rust remains the authority for typed semantics."""
import os
from pathlib import Path
from .context import context_settings
from .eider_direct import EiderBridge, EiderDecision, _LOCK


def validate(body):
    if not isinstance(body,dict) or not {'state','questions'} <= set(body) or set(body)-{'state','questions','model','media'}:
        raise ValueError('Expected state/questions and optional model/media')
    if not isinstance(body['state'],str) or not body['state'] or len(body['state'])>(200000 if 'media' in body else 2000000):
        raise ValueError('Public media adapter requires bounded nonempty text state')
    if 'media' in body:
        from .media import validate_media
        validate_media(body['media'])
    from .model import MODEL
    if 'model' in body and body['model'] not in ('local-gemma4',MODEL['model']):raise ValueError('Requested model is not the served NVFP4 Gemma checkpoint')
    qs=body['questions']
    if not isinstance(qs,dict) or not 1<=len(qs)<=64:raise ValueError('Expected 1..64 questions')
    for key,q in qs.items():
        if not isinstance(key,str) or not key or len(key)>128:raise ValueError('Invalid question id')
        if not isinstance(q,dict) or not {'type','instructions'}<=set(q) or set(q)-{'type','instructions','criteria'}:raise ValueError('Invalid question')
        if not isinstance(q['instructions'],(str,dict,list)):raise ValueError('Invalid instructions')
        kind=q['type']; c=q.get('criteria')
        if kind=='choice':
            if not isinstance(c,dict) or not 2<=len(c)<=64 or any(not isinstance(k,str) or not k or (v is not None and not isinstance(v,str)) for k,v in c.items()):raise ValueError('Invalid choice criteria')
        elif kind=='score':
            if not isinstance(c,list) or not 2<=len(c)<=64 or any(not isinstance(v,str) for v in c):raise ValueError('Invalid score criteria')
        elif kind=='noul':
            if c is not None and (not isinstance(c,dict) or set(c)-{'true','false'} or any(not isinstance(v,str) for v in c.values())):raise ValueError('Invalid noul criteria')
            if 'criteria' in q and c is None:raise ValueError('Null criteria')
        else:raise ValueError('Expected choice, noul or score')
    return body


class EiderEngine(EiderDecision):
    def __init__(self,model_path,max_input_tokens=None,media=False,bridge_library=None,hardware="auto"):
        if not __debug__:raise RuntimeError("Python -O disables required runtime guards; use normal Python")
        with _LOCK:
            limit,context,kv=context_settings(max_input_tokens,media)
            library=bridge_library or os.environ.get('GEMMA_EIDER_LIBRARY')
            if not library or not Path(library).is_file():raise ValueError('Set GEMMA_EIDER_LIBRARY to the built Eider bridge library; see docs/EIDER.md')
            from .backends import _LOADED
            if _LOADED:raise RuntimeError('One backend/bridge per process; restart to change configuration')
            bridge=EiderBridge(library,model_path)
            from .backends import load_backend
            backend=load_backend(model_path,media=media,context=context,kv_bytes=None if not media and context == 65536 else kv,hardware=hardware)
            super().__init__(bridge,backend,limit+1)
