# SPDX-License-Identifier: Apache-2.0
# Runtime adaptations of vLLM launch/reduction code; see NOTICE.
import hashlib, inspect
from pathlib import Path
TRIAL = dict(query_block=32, attention_warps=4, attention_stages=1, sliding_stages=1, moe_sum_out=True, moe_alias=True, compact_head=True)
def prepare_attention(trial, audit):
    if trial.get('query_block'):
        from vllm.v1.attention.ops import triton_unified_attention as attention
        original_launch = attention.unified_attention
        source = inspect.getsource(original_launch)
        target = '    BLOCK_Q = BLOCK_M // num_queries_per_kv\n'
        # One ordinary launch calculation and one existing SM100 tuning branch.
        assert source.count(target) == 2
        assert trial['query_block'] in (32, 64)
        replacement = ('    if head_size == 512 and window_size[0] < 0 and max_seqlen_q > 1:\n'
                       f"        BLOCK_M = {trial['query_block']}\n" + target)
        modified = source.replace(target, replacement, 1)
        namespace = {}
        exec(compile(modified, '<wi033-existing-attention-launch>', 'exec'),
             original_launch.__globals__, namespace)
        original_launch.__code__ = namespace['unified_attention'].__code__
        audit['patches'].append({'name': 'existing_full_attention_query_block',
                                 'value': trial['query_block'], 'original_host_function': source,
                                 'modified_host_sha256': hashlib.sha256(modified.encode()).hexdigest(),
                                 'original_host_sha256': hashlib.sha256(source.encode()).hexdigest(),
                                 'kernel_body_changed': False})
    if 'attention_warps' in trial:
        from vllm.v1.attention.ops import triton_unified_attention as attention
        original = attention.unified_attention
        source = next((p['original_host_function'] for p in audit['patches'] if p['name'] == 'existing_full_attention_query_block'), None)
        if source is None:
            source = inspect.getsource(original)
        # Combine with the prior host-only query32 override; inspect.getsource sees original file.
        target = '    BLOCK_Q = BLOCK_M // num_queries_per_kv\n'
        if trial.get('query_block'):
            source = source.replace(target, f'    if head_size == 512 and window_size[0] < 0 and max_seqlen_q > 1:\n        BLOCK_M = {trial["query_block"]}\n' + target, 1)
        target = '    kernel_unified_attention[grid](\n'
        assert source.count(target) == 1
        assert trial['attention_warps'] in (4, 8) and trial['attention_stages'] in (1, 2)
        replacement = ('    if head_size == 512 and window_size[0] < 0 and max_seqlen_q > 1:\n'
                       f"        launch_kwargs['num_warps'] = {trial['attention_warps']}\n"
                       f"        launch_kwargs['num_stages'] = {trial['attention_stages']}\n" + target)
        if trial.get('sliding_stages'):
            assert trial['sliding_stages'] in (1, 2)
            replacement = replacement.replace(target,
                "    if head_size == 256 and window_size[0] >= 0 and max_seqlen_q > 1:\n"
                "        launch_kwargs['num_warps'] = 4\n"
                f"        launch_kwargs['num_stages'] = {trial['sliding_stages']}\n" + target)
        modified = source.replace(target, replacement)
        namespace = {}
        exec(compile(modified, '<wi041-existing-launch-options>', 'exec'), original.__globals__, namespace)
        original.__code__ = namespace['unified_attention'].__code__
        audit['patches'].append({'name':'full_attention_launch_options', 'warps':trial['attention_warps'],
                                'stages':trial['attention_stages'], 'modified_source':modified,
                                'modified_sha256':hashlib.sha256(modified.encode()).hexdigest(), 'kernel_body_changed':False})
    if trial.get('tile'):
        from vllm.v1.attention.ops import triton_unified_attention as attention
        original = attention._get_tile_size
        tile = trial['tile']
        assert tile in (16, 64, 128)

        def select(head_size, sliding_window, element_size, is_prefill):
            if head_size == 512 and sliding_window == 0 and is_prefill:
                return tile
            return original(head_size, sliding_window, element_size, is_prefill)

        attention._get_tile_size = select
        audit['patches'].append({'name': 'existing_full_attention_prefill_tile',
                                 'value': tile, 'original': inspect.getsource(original),
                                 'kernel_source_sha256': hashlib.sha256(
                                     Path(attention.__file__).read_bytes()).hexdigest()})

def install():
    trial=TRIAL
    audit={"patches": []}
    prepare_attention(trial,audit)
    if trial.get('moe_sum_out'):
        import textwrap
        from vllm.model_executor.layers.fused_moe.experts import cutlass_moe
        original = cutlass_moe.run_cutlass_moe_fp4
        source = inspect.getsource(original)
        target = """        output.copy_(
            (
                c3.view(m, num_topk, k)
                * topk_weights.view(m, num_topk, 1).to(out_dtype)
            ).sum(dim=1),
            non_blocking=True,
        )"""
        replacement = """        assert output.is_contiguous()
        torch.sum(
            c3.view(m, num_topk, k)
            * topk_weights.view(m, num_topk, 1).to(out_dtype),
            dim=1, out=output,
        )"""
        if trial.get('moe_inplace'):
            replacement = """        assert output.is_contiguous()
        assert c3.is_contiguous() and c3.dtype == out_dtype
        assert c3.untyped_storage().data_ptr() != output.untyped_storage().data_ptr()
        assert c3.untyped_storage().data_ptr() != workspace13.untyped_storage().data_ptr()
        c3.view(m, num_topk, k).mul_(topk_weights.view(m, num_topk, 1).to(out_dtype))
        torch.sum(c3.view(m, num_topk, k), dim=1, out=output)"""
        assert source.count(target) == 1
        modified = source.replace(target, replacement)
        namespace = {}
        exec(compile(modified, '<wi041-existing-sum-output>', 'exec'), original.__globals__, namespace)
        original.__code__ = namespace['run_cutlass_moe_fp4'].__code__
        audit['patches'].append({'name':'moe_existing_sum_out', 'original_source':source,
                                'modified_source':modified, 'kernel_body_changed':False,
                                'modified_sha256':hashlib.sha256(modified.encode()).hexdigest()})
    if trial.get('moe_alias'):
        import textwrap
        from vllm.model_executor.layers.fused_moe import modular_kernel
        original = modular_kernel.FusedMoEKernelModularImpl._fused_experts
        source = textwrap.dedent(inspect.getsource(original))
        target = '    self.fused_experts.apply(\n'
        replacement = """    assert type(self.fused_experts).__name__ == 'CutlassExpertsFp4'
    assert not self.prepare_finalize.supports_async() and not dbo_enabled()
    assert getattr(self.fused_experts, '_lora_context', None) is None
    assert type(self.fused_experts.finalize_weight_and_reduce_impl()).__name__ == 'TopKWeightAndReduceNoOP'
    assert output_alias is not None and output_alias.is_contiguous()
    assert output_alias.shape == fused_out.shape and output_alias.dtype == fused_out.dtype
    assert output_alias.device == fused_out.device
    for other in (a1q, workspace13, workspace2, topk_weights, w1, w2):
        assert output_alias.untyped_storage().data_ptr() != other.untyped_storage().data_ptr()
    fused_out = output_alias
""" + target
        assert source.count(target) == 1
        modified = source.replace(target, replacement)
        namespace = {}
        exec(compile(modified, '<wi041-scoped-output-alias>', 'exec'), original.__globals__, namespace)
        original.__code__ = namespace['_fused_experts'].__code__
        audit['patches'].append({'name':'scoped_fp4_output_alias','original_source':source,
                                'modified_source':modified,'modified_sha256':hashlib.sha256(modified.encode()).hexdigest()})
    return [p["name"] for p in audit["patches"]]
