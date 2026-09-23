# SPDX-License-Identifier: Apache-2.0 AND MIT
# Aligned slicing follows ExLlamaV3 LinearEXL3.tp_import_split_n (MIT).
# Modified to select ABC blocks in a single-device decision head. See NOTICE.
def compact_head(model,torch,ids,retain_base=True):
 from exllamav3.modules.quant.exl3 import LinearEXL3
 head=next(m for m in model.modules if m.caps.get('logits_output'))
 full=head.inner;assert isinstance(full,LinearEXL3)
 assert not head.trim_padded_out and not head.lora_a_tensors
 blocks=sorted(set(i//128 for i in ids));mapped=[blocks.index(i//128)*128+i%128 for i in ids]
 compact=LinearEXL3(config=full.config,in_features=full.in_features,out_features=128*len(blocks),suh=full.suh,
  svh=torch.cat([full.svh[b*128:(b+1)*128] for b in blocks]),
  trellis=torch.cat([full.trellis[:,b*8:(b+1)*8,:] for b in blocks],dim=1),
  mcg=full.mcg_tensor,mul1=full.mul1_tensor,
  bias=None if full.bias is None else torch.cat([full.bias[b*128:(b+1)*128] for b in blocks]),out_dtype=full.out_dtype,key='wi051_compact_head')
 original=head.forward
 def selected(x,params,*args,**kwargs):
  y=original(x,params,*args,**kwargs)
  return y[...,mapped] if head.inner is compact else y
 head.forward=selected
 audit={'ids':ids,'blocks':blocks,'mapped_ids':mapped,'original_outputs':full.out_features,'compact_outputs':compact.out_features,'head_bits':full.K,'softcap':head.softcap,'pre_scale':head.pre_scale,'post_scale':head.post_scale,'retains_full_for_pairing':retain_base}
 if retain_base:
  def switch(variant):head.inner=full if variant=='base' else compact
 else:
  head.inner=compact
  def switch(variant):assert variant=='candidate'
 return switch,audit
