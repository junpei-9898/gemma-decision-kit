"""User-supplied request benchmark; no self-grading; media mode has bounded prefix/processor caching."""
import argparse,json,statistics,time
from contextlib import redirect_stdout
import sys
from gemma_decision.eider_engine import EiderEngine
from gemma_decision.cli import parse_json
p=argparse.ArgumentParser();p.add_argument('--media',action='store_true');p.add_argument('--model-path',required=True);p.add_argument('--input',required=True);p.add_argument('--repeats',type=int,default=10);a=p.parse_args()
if not 1<=a.repeats<=100:p.error('repeats must be1..100')
body=parse_json(open(a.input,encoding='utf8').read());times=[]
with redirect_stdout(sys.stderr):
 engine=EiderEngine(a.model_path,media=a.media);engine.predict(body)
 for _ in range(a.repeats):
  start=time.perf_counter();result=engine.predict(body);times.append((time.perf_counter()-start)*1000)
print(json.dumps({'semantics':'eider-decision-v1','repeats':a.repeats,'questions':len(body['questions']),'input_tokens':result['usage']['input_tokens'],'median_ms':statistics.median(times),'min_ms':min(times),'max_ms':max(times),'accuracy':'not evaluated'},indent=2))
