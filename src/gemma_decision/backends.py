# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from .profiles import PROFILES
_LOADED=False


def load_backend(profile, model_path):
    global _LOADED
    if _LOADED:raise RuntimeError('One backend per process; restart to change profile')
    if not Path(model_path).is_dir():raise ValueError('A local model directory is required')
    _LOADED=True
    return SpeedBackend(model_path) if profile=='speed' else MemoryBackend(model_path)


def candidate_ids(tokenizer):
    encoded=[tokenizer.encode(c,add_special_tokens=False) for c in 'ABC']
    if any(len(x)!=1 for x in encoded):raise RuntimeError('ABC must be single tokens')
    return [x[0] for x in encoded]


class SpeedBackend:
    def __init__(self,path):
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
        import vllm
        if vllm.__version__!='0.26.1.dev0+gf2654939e.d20260726':raise RuntimeError('speed requires the exact documented GB10 image (vLLM 0.26.1.dev0+gf2654939e.d20260726)')
        from vllm import LLM, SamplingParams
        from .nv_scale import install as install_scale
        from .nv_runtime import install as install_runtime, TRIAL
        from .nv_head import install_candidates
        install_scale();install_runtime()
        self.model=LLM(model=path,trust_remote_code=False,dtype='auto',kv_cache_dtype='fp8_e4m3',max_model_len=65536,kv_cache_memory_bytes=3221225472,gpu_memory_utilization=.25,max_num_seqs=1,max_num_batched_tokens=8192,enable_prefix_caching=False,enforce_eager=True,async_scheduling=False,logprobs_mode='processed_logprobs',limit_mm_per_prompt={'image':0,'audio':0,'video':0},seed=0,kernel_config={'moe_backend':'cutlass'})
        self.tokenizer=self.model.get_tokenizer();self.ids=candidate_ids(self.tokenizer)
        self.model.apply_model(lambda m:install_candidates(m,TRIAL,self.ids,None))
        self.params=SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=1,allowed_token_ids=self.ids,logprobs=3,seed=0)

    def score(self,tokens):
        import math
        r=self.model.generate([{'prompt_token_ids':tokens}],self.params,use_tqdm=False)[0]
        if r.prompt_token_ids!=tokens:raise RuntimeError('Backend changed input tokens')
        return [math.exp(r.outputs[0].logprobs[0][i].logprob) for i in self.ids]


class MemoryBackend:
    def __init__(self,path):
        import torch
        from transformers import AutoTokenizer
        from .runtime_check import verify_exl_source
        verify_exl_source()
        from exllamav3 import Config,Model
        from .exl_head import compact_head
        # Override experimental knobs before importing MoE modules via exllamav3.
        if os.environ.get('EXL3_MOE_FUSED_ROWS','128')!='128' or os.environ.get('EXL3_MOE_FUSED_DET','1')!='1':
            raise RuntimeError('Only default MoE128 and deterministic accumulation are supported')
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=False,local_files_only=True)
        self.ids=candidate_ids(self.tokenizer)
        cfg=Config.from_directory(path);self.model=Model.from_config(cfg)
        self.model.load(device=torch.device('cuda:0'),progressbar=False)
        switch,self.audit=compact_head(self.model,torch,self.ids,retain_base=False);switch('candidate')

    def score(self,tokens):
        x=self.torch.tensor([tokens],dtype=self.torch.long)
        logits=self.model.forward(x,{'attn_mode':'flash_attn_nc','causal':True,'last_tokens_only':1})
        return self.torch.softmax(logits[0,0].float(),dim=-1).cpu().tolist()
