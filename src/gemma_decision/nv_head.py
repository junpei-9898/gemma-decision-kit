# SPDX-License-Identifier: Apache-2.0
import inspect, types
def install_candidates(model, trial, ids, out):
    import torch
    audit={'patches':[], 'quant_modules':[]}
    for name,m in model.named_modules():
        q=getattr(m,'quant_method',None)
        if q is not None:
            audit['quant_modules'].append({'name':name,'class':type(m).__name__,'quant':type(q).__name__,'kernel':type(getattr(q,'kernel',None)).__name__})
    if trial.get('compact_head'):
        processors=[m for m in model.modules() if type(m).__name__=='LogitsProcessor']
        assert len(processors)==1
        processor=processors[0]
        assert processor.soft_cap==30 and processor.scale==1 and processor.head_dtype in [None,torch.bfloat16]
        head=next(m for m in model.modules() if type(m).__name__=='ParallelLMHead')
        assert type(head.quant_method).__name__=='UnquantizedLinearMethod'
        assert head.weight.shape[0]==262144 and head.weight.dtype==torch.bfloat16
        candidate_weight=head.weight[ids].detach().contiguous()
        source=inspect.getsource(processor._get_logits)
        def compact(self, hidden_states, lm_head, embedding_bias=None):
            assert lm_head is head and embedding_bias is None
            values=torch.nn.functional.linear(hidden_states,candidate_weight)
            result=torch.full((*values.shape[:-1],self.vocab_size),float('-inf'),dtype=values.dtype,device=values.device)
            result[...,ids]=values
            return result
        processor._get_logits=types.MethodType(compact,processor)
        audit['patches'].append({'name':'compact_abc_projection','ids':ids,'weight_shape':list(candidate_weight.shape),'original_source':source,'softcap_preserved':30,'allowed_mask_still_required':True})
    return {'patches':len(audit['patches'])}
