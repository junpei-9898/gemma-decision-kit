# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from .profiles import PROFILES
_LOADED=False


def load_backend(profile, model_path, media=False, context=65536, kv_bytes=3221225472):
    global _LOADED
    if _LOADED:raise RuntimeError('One backend per process; restart to change profile')
    if not Path(model_path).is_dir():raise ValueError('A local model directory is required')
    _LOADED=True
    if profile != "speed":raise ValueError("Only NVFP4 speed profile is distributed")
    return SpeedBackend(model_path, media=media, context=context, kv_bytes=kv_bytes)


def candidate_ids(tokenizer):
    encoded=[tokenizer.encode(c,add_special_tokens=False) for c in 'ABC']
    if any(len(x)!=1 for x in encoded):raise RuntimeError('ABC must be single tokens')
    return [x[0] for x in encoded]


class SpeedBackend:
    def __init__(self,path,media=False,context=65536,kv_bytes=3221225472):
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
        import vllm
        if vllm.__version__!='0.26.1.dev0+gf2654939e.d20260726':raise RuntimeError('speed requires the exact documented GB10 image (vLLM 0.26.1.dev0+gf2654939e.d20260726)')
        from vllm import LLM, SamplingParams
        from .nv_scale import install as install_scale
        from .nv_runtime import install as install_runtime, TRIAL
        from .nv_head import install_candidates
        install_scale()
        self.media=media
        if media:
            from .nv_runtime import prepare_attention
            prepare_attention({'query_block':32},{'patches':[]})
        else:install_runtime()
        self.model=LLM(model=path,trust_remote_code=False,dtype='auto',kv_cache_dtype='auto' if media else 'fp8_e4m3',max_model_len=context,kv_cache_memory_bytes=kv_bytes,gpu_memory_utilization=.25,max_num_seqs=1,max_num_batched_tokens=8192,enable_prefix_caching=media,enforce_eager=True,async_scheduling=False,logprobs_mode='processed_logprobs',limit_mm_per_prompt={'image':int(media),'audio':0,'video':int(media)},mm_processor_cache_gb=.125 if media else 0,seed=0,kernel_config={'moe_backend':'cutlass'})
        self.tokenizer=self.model.get_tokenizer();self.ids=candidate_ids(self.tokenizer)
        if not media:self.model.apply_model(lambda m:install_candidates(m,TRIAL,self.ids,None))
        self.params=SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=1,allowed_token_ids=self.ids,logprobs=3,seed=0)

    def score(self,tokens):
        import math
        r=self.model.generate([{'prompt_token_ids':tokens}],self.params,use_tqdm=False)[0]
        if r.prompt_token_ids!=tokens:raise RuntimeError('Backend changed input tokens')
        return [math.exp(r.outputs[0].logprobs[0][i].logprob) for i in self.ids]

    def prepare_media(self, messages, limit):
        if not self.media:raise ValueError('Restart with --media for image/video inputs')
        # Pinned vLLM preprocessing; inspect full expanded tokens before enqueue.
        prepared=self.model._preprocess_chat_one(messages,chat_template_kwargs={'enable_thinking':False})
        ids=prepared['prompt_token_ids']
        if len(ids)>limit:raise ValueError('Media question exceeds expanded token limit; input was not truncated')
        cfg=self.model.model_config.hf_config
        counts={name:ids.count(getattr(cfg,field,None)) for name,field in [('image','image_token_id'),('video','video_token_id')]}
        if not any(counts.values()):raise RuntimeError('No media tokens after preprocessing')
        return prepared,ids,counts

    def score_media(self, prepared):
        import math
        from vllm.outputs import RequestOutput
        self.model._add_request(prepared,self.params)
        r=self.model._run_engine(output_type=RequestOutput,use_tqdm=False)[0]
        if r.prompt_token_ids!=prepared['prompt_token_ids']:raise RuntimeError('Backend changed media tokens')
        return [math.exp(r.outputs[0].logprobs[0][i].logprob) for i in self.ids]
