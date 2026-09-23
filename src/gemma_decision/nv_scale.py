# SPDX-License-Identifier: Apache-2.0
# Adapted from the vLLM gate/up scale reconciliation approach; see NOTICE.
def reconcile(block, global_scale):
    """Upstream 9ac09fb algorithm: compensate block scales before sharing global."""
    import torch
    if torch.allclose(global_scale[:, 0], global_scale[:, 1]):
        return block, global_scale[:, 0].contiguous()
    common = torch.maximum(global_scale[:, 0], global_scale[:, 1])
    factors = global_scale / common.unsqueeze(1)
    n, fused, width = block.shape
    assert fused % 2 == 0 and torch.isfinite(common).all() and (common > 0).all()
    result = block.reshape(n, 2, fused // 2, width).float()
    result *= factors[:, :, None, None]
    return result.to(block.dtype).reshape(block.shape).contiguous(), common.contiguous()

def install():
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE
    original = ModelOptNvFp4FusedMoE.process_weights_after_loading
    def process(self, layer):
        fixed, common = reconcile(layer.w13_weight_scale, layer.w13_weight_scale_2)
        layer.w13_weight_scale.data.copy_(fixed)
        layer.w13_weight_scale_2.data.copy_(common[:, None].expand_as(layer.w13_weight_scale_2))
        return original(self, layer)
    ModelOptNvFp4FusedMoE.process_weights_after_loading = process
