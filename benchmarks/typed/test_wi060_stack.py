# SPDX-License-Identifier: Apache-2.0
# @unit evaluation
# @layer infrastructure
# @work-item-id WI-060
"""Own loopback servers only; terminate own child processes gracefully."""
import sys,os,subprocess,json,time,urllib.request,signal
from pathlib import Path
W=Path('/work')
def main():
    name=sys.argv[1];model=name.split('-r')[0];out=W/('stack-'+name);out.mkdir(exist_ok=False);children=[]
    def shutdown(*_):
        for p in reversed(children):
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
    def interrupted(*_):
        shutdown();raise SystemExit(130)
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    def launch(args,log):
        (out/(log+'.command.json')).write_text(json.dumps(args,indent=2));f=(out/log).open('x');p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append(p);return p
    def ready(url,p):
        start=time.monotonic()
        while time.monotonic()-start<900:
            if p.poll() is not None:raise RuntimeError('owned server exited')
            try:
                with urllib.request.urlopen(url,timeout=2) as r:
                    if r.status==200:return
            except Exception:pass
            time.sleep(2)
        raise RuntimeError('server readiness timeout')
    try:
        if model=='qwen':
            args=['/work/eider-serve','--model-dir','/work/imported/models/qwen-eider','--artifact-dir','/work/imported/artifacts','--offline','--listen','127.0.0.1:18080','--served-model-name','local-qwen3_5_moe','--max-context-tokens','8192','--max-active-sequences','2','--decision-branch-capacity','1','--prefill-sequence-capacity','1','--prefill-token-capacity','512','--decode-capacity','1','--speculative-drafts','0','--retained-prefix-gib','0']
            p=launch(args,'server.log');ready('http://127.0.0.1:18080/v1/models',p)
        else:
            args=[sys.executable,'-m','vllm.entrypoints.cli.main','serve','/work/assets/models/diffusiongemma','--served-model-name','diffusiongemma','--host','127.0.0.1','--port','18090','--diffusion-config','{"canvas_length":64}','--max-logprobs','32','--no-enable-prefix-caching','--async-scheduling','--attention-backend','TRITON_ATTN','--max-num-seqs','1','--kv-cache-memory-bytes','4294967296','--max-model-len','8192','--max-num-batched-tokens','2048','--gpu-memory-utilization','0.50','--limit-mm-per-prompt','{"image":0,"audio":0,"video":0}']
            p=launch(args,'server.log');ready('http://127.0.0.1:18090/health',p)
            source=next((W/'assets/upstream').glob('vllm-*'))
            p=launch([sys.executable,str(source/'examples/features/diffusion_reads/structured_server.py'),'--upstream','http://127.0.0.1:18090','--model','diffusiongemma','--tokenizer','/work/assets/models/diffusiongemma','--canvas','64','--host','127.0.0.1','--port','18091'],'structured.log')
            import socket
            for _ in range(60):
                if p.poll() is not None:raise RuntimeError('structured server exited')
                try:
                    with socket.create_connection(('127.0.0.1',18091),timeout=1):break
                except OSError:time.sleep(1)
            else:raise RuntimeError('listener timeout')
        p=launch([sys.executable,'/work/test_wi060_runner.py',name],'evaluation.log');p.wait();assert p.returncode==0
    finally:
        shutdown()
        for p in reversed(children):
            try:p.wait(timeout=60)
            except subprocess.TimeoutExpired:raise RuntimeError('owned process remains; supervisor must verify')
if __name__=='__main__':main()
