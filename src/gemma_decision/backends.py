# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
_LOADED=False


def load_backend(model_path, media=False, context=65536, kv_bytes=None, hardware="auto"):
    global _LOADED
    if _LOADED:raise RuntimeError('One backend per process; restart to change configuration')
    if not Path(model_path).is_dir():raise ValueError('A local model directory is required')
    _LOADED=True
    return SpeedBackend(model_path, media=media, context=context, kv_bytes=kv_bytes, hardware=hardware)


def candidate_ids(tokenizer):
    encoded=[tokenizer.encode(c,add_special_tokens=False) for c in 'ABC']
    if any(len(x)!=1 for x in encoded):raise RuntimeError('ABC must be single tokens')
    return [x[0] for x in encoded]


class SpeedBackend:
    def __init__(self,path,media=False,context=65536,kv_bytes=None,decision_kv_dtype=None,hardware="auto"):
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
        import vllm
        from .hardware import detect_hardware,validate_runtime
        self.hardware=detect_hardware(hardware)
        validate_runtime(vllm.__version__,self.hardware)
        from vllm import LLM
        from .nv_scale import install as install_scale
        from .nv_runtime import install as install_runtime, TRIAL
        from .nv_head import install_candidates
        install_scale()
        self.media=media
        self.decision_logits=True
        if self.hardware=='spark' and media:
            from .nv_runtime import prepare_attention
            prepare_attention({'query_block':32},{'patches':[]})
        elif self.hardware=='spark':install_runtime()
        if kv_bytes is None:
            kv_bytes = 1744830464 if self.hardware == 'spark' and not media and context == 65536 else 3 * 1024**3
        self.model=LLM(model=path,trust_remote_code=False,dtype='auto',kv_cache_dtype=decision_kv_dtype or ('auto' if media else 'fp8_e4m3'),max_model_len=context,kv_cache_memory_bytes=kv_bytes,gpu_memory_utilization=.25,max_num_seqs=1,max_num_batched_tokens=8192,enable_prefix_caching=media,enforce_eager=True,async_scheduling=False,logprobs_mode='raw_logits',max_logprobs=64,limit_mm_per_prompt={'image':int(media),'audio':0,'video':int(media)},mm_processor_cache_gb=.125 if media else 0,seed=0,kernel_config={'moe_backend':'cutlass'})
        self.tokenizer=self.model.get_tokenizer();self.ids=candidate_ids(self.tokenizer)
        if self.hardware=='spark' and not media:self.model.apply_model(lambda m:install_candidates(m,TRIAL,self.ids,None))

    def selected_logits(self, tokens, ids):
        from .nv_head import install_candidates
        from .nv_runtime import TRIAL
        from vllm import SamplingParams
        import math
        if self.hardware=='spark' and not self.media and self.ids != ids:
            self.model.apply_model(lambda m:install_candidates(m,TRIAL,ids,None))
            self.ids=list(ids)
        params=SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=1,allowed_token_ids=ids,logprobs=len(ids),logprob_token_ids=ids if self.media or self.hardware=='standard' else None,seed=0)
        r=self.model.generate([{'prompt_token_ids':tokens}],params,use_tqdm=False)[0]
        if r.prompt_token_ids!=tokens:raise RuntimeError('Backend changed input tokens')
        values=[r.outputs[0].logprobs[0][i].logprob for i in ids]
        if not all(math.isfinite(v) for v in values):raise RuntimeError('Nonfinite selected logits')
        return values

    def prepare_eider_media(self, item, tokens, marker, limit, suffix):
        import json
        from .media import messages
        if not self.media:raise ValueError('Restart with --media')
        text=self.tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if self.tokenizer.encode(text,add_special_tokens=False)!=tokens:
            raise RuntimeError('Eider token roundtrip changed; media adapter refuses re-tokenization')
        if text.count(marker)!=1:raise RuntimeError('Media marker must occur exactly once')
        before,after=text.split(marker)
        # Literal JSON strings, never template interpolation of user content.
        # The pinned Gemma4 checkpoint template emits <|image|>/<|video|>.
        # Its string-mode fallback emits image_soft_token, which is not a processor input marker.
        visual_marker='<|image|>' if item['type']=='image' else '<|video|>'
        template='{{ '+json.dumps(before+visual_marker+after)+' }}'
        conversation=messages(item,'')
        conversation[0]['content']=conversation[0]['content'][:1]
        prepared=self.model._preprocess_chat_one(conversation,chat_template=template,
            chat_template_content_format='openai',add_generation_prompt=False,
            tokenization_kwargs={'add_special_tokens':False})
        ids=prepared['prompt_token_ids']
        if len(ids)>limit:raise ValueError('Expanded media token limit exceeded; no truncation')
        if ids[-len(suffix):]!=suffix:raise RuntimeError('Media processor changed Eider question suffix')
        cfg=self.model.model_config.hf_config
        counts={name:ids.count(getattr(cfg,field,None)) for name,field in [('image','image_token_id'),('video','video_token_id')]}
        if not any(counts.values()):raise RuntimeError('Missing visual tokens')
        if not prepared.get('mm_kwargs') and not prepared.get('multi_modal_data'):
            raise RuntimeError('Missing visual features')
        return prepared,ids,counts

    def selected_media_logits(self, prepared, ids):
        import math
        from vllm import SamplingParams
        from vllm.outputs import RequestOutput
        if not self.decision_logits or not self.media:raise RuntimeError('Requires Eider media backend')
        params=SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=1,allowed_token_ids=ids,logprobs=len(ids),logprob_token_ids=ids if self.media or self.hardware=='standard' else None,seed=0)
        self.model._add_request(prepared,params)
        r=self.model._run_engine(output_type=RequestOutput,use_tqdm=False)[0]
        if r.prompt_token_ids!=prepared['prompt_token_ids']:raise RuntimeError('Media tokens changed')
        values=[r.outputs[0].logprobs[0][i].logprob for i in ids]
        if not all(math.isfinite(v) for v in values):raise RuntimeError('Nonfinite selected logits')
        return values
