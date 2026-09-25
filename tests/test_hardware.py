# SPDX-License-Identifier: Apache-2.0
import unittest
from gemma_decision.hardware import select_hardware,validate_runtime,PINNED_VLLM
class HardwareTests(unittest.TestCase):
    def select(self,requested='auto',**kwargs):
        options=dict(capability=(12,1),name='NVIDIA GB10',machine='aarch64',system='Linux');options.update(kwargs)
        return select_hardware(requested,**options)
    def test_gb10_auto_and_standard_opt_in(self):
        self.assertEqual(self.select('spark'),'spark');self.assertEqual(self.select(),'spark');self.assertEqual(self.select('standard'),'standard')
    def test_other_blackwell_gets_no_gb10_patches(self):
        for name,cc in [('RTX 5090',(12,0)),('NVIDIA B200',(10,0))]:
            self.assertEqual(self.select(name=name,capability=cc,machine='x86_64'),'standard')
            with self.assertRaises(ValueError):self.select('spark',name=name,capability=cc,machine='x86_64')
    def test_fail_closed_unsupported(self):
        for opts in [dict(capability=(8,9)),dict(system='Darwin'),dict(system='Windows')]:
            with self.assertRaises(ValueError):self.select(**opts)
        with self.assertRaises(ValueError):self.select('spark',machine='x86_64')
        with self.assertRaises(ValueError):self.select('gb10',machine='x86_64')
    def test_standard_never_installs_gb10_runtime_or_head(self):
        import types,os
        from unittest.mock import Mock,patch
        from gemma_decision.backends import SpeedBackend
        for profile,media in [('standard',False),('standard',True),('spark',False),('spark',True)]:
            model=Mock();model.get_tokenizer.return_value.encode.side_effect=lambda c,**kw:[ord(c)]
            module=types.SimpleNamespace(__version__=PINNED_VLLM,LLM=Mock(return_value=model),SamplingParams=Mock())
            with patch.dict('sys.modules',{'vllm':module}),patch.dict(os.environ,{}),patch('gemma_decision.hardware.detect_hardware',return_value=profile),patch('gemma_decision.nv_scale.install') as scale,patch('gemma_decision.nv_runtime.install') as runtime,patch('gemma_decision.nv_runtime.prepare_attention') as attention:
                SpeedBackend('/unused',media=media,hardware=profile)
                scale.assert_called_once()
                self.assertEqual(runtime.call_count,int(profile=='spark' and not media))
                self.assertEqual(attention.call_count,int(profile=='spark' and media))
                self.assertEqual(model.apply_model.call_count,int(profile=='spark' and not media))
    def test_runtime_pin(self):
        for hardware in ['standard','spark']:validate_runtime(PINNED_VLLM,hardware)
        validate_runtime('0.26.1.dev0+gf2654939e.d20260925','standard')
        with self.assertRaises(RuntimeError):validate_runtime('0.26.1.dev0+gf2654939e.d20260925','spark')
        with self.assertRaises(RuntimeError):validate_runtime('0.27.0','standard')
if __name__=='__main__':unittest.main()
