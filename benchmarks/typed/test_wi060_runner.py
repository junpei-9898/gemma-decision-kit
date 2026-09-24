# SPDX-License-Identifier: Apache-2.0
# @unit evaluation
# @layer infrastructure
# @work-item-id WI-060
"""Execute frozen typed requests with no gold labels available to the predictor."""
import sys,json,time,hashlib,statistics
from pathlib import Path
from test_wi060_adapters import Gemma,Direct,Http
W=Path('/work')
def group(c):
    q=c['body']['questions']['q'];return 'choice'+str(len(q['criteria'])) if q['type']=='choice' else q['type']
def emit(p,r):
    with p.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def main():
    name=sys.argv[1];model=name.split('-r')[0];out=W/('result-'+name);out.mkdir(exist_ok=False)
    raw=(W/'cases.json').read_bytes();cases=json.loads(raw);groups=['choice2','choice4','choice8','noul','score']
    a=Gemma() if model=='gemma' else Direct(model) if model in ['semif','laya','nanojev'] else Http(model)
    selected={g:[c for c in cases if group(c)==g][:3] for g in groups}
    plan=[('warmup',0,selected[g][0]) for g in groups]+[('quality',0,c) for c in cases]
    plan += [('latency',rep,c) for g in groups for rep in range(5) for c in selected[g]]
    # Eight queries on one shared state; no concatenation of unrelated states.
    state='単価120円の商品8個を購入。合計は960円。全4項目の検品が完了。荷物は未発送。見積ではなく返金を希望。'
    template=[('noul','荷物は発送済みです。',{'true':'正しい','false':'誤り'}),('choice','主な依頼は何か。',{'refund':'返金','repair':'修理'}),('score','完了した検品数。',['0項目','1項目','2項目','3項目','4項目']),('choice','購入個数は。',{'six':'6個','seven':'7個','eight':'8個','nine':'9個'}),('noul','合計は960円です。',{'true':'正しい','false':'誤り'}),('choice','荷物の状態は。',{'unsent':'未発送','shipped':'発送済み'}),('score','完了した検品数。',['0項目','1項目','2項目','3項目','4項目']),('choice','商品単価は。',{'a':'100円','b':'110円','c':'120円','d':'130円','e':'140円','f':'150円','g':'160円','h':'170円'})]
    mixed={'id':'mixed8','body':{'state':state,'questions':{f'q{i}':dict(type=t,instructions=ins,criteria=crit) for i,(t,ins,crit) in enumerate(template)}}}
    # Warm this exact workflow once; all subsequent calls retained.
    plan += [('mixed-warmup',0,mixed)]+[('mixed-latency',i,mixed) for i in range(5)]
    total_tokens=0;sequences=0;failures=0
    import torch
    sync=lambda:torch.cuda.synchronize() if model not in ['qwen','diffusion'] else None
    for index,(phase,rep,c) in enumerate(plan):
        if (W/('STOP-'+name)).exists():raise RuntimeError('supervisor safety stop')
        b=c['body'];sequences+=len(b['questions']);assert sequences<=500
        emit(out/'ledger.jsonl',{'event':'planned','index':index,'case':c['id'],'sequences':len(b['questions']),'time':time.time()})
        sync();start=time.perf_counter();r=None;status='PASS'
        try:r=a.predict(b);sync();elapsed=time.perf_counter()-start
        except Exception as exc:
            elapsed=time.perf_counter()-start;status='FAIL';r={'error':repr(exc),'traceback':__import__('traceback').format_exc()};failures+=1
        coverage=a.coverage(b) if status=='PASS' else None
        row={'case':c['id'],'group':group(c) if c['id']!='mixed8' else 'mixed8','phase':phase,'repeat':rep,'status':status,'seconds':elapsed,'response':r,'coverage':coverage,'request_sha256':hashlib.sha256(json.dumps(b,ensure_ascii=False).encode()).hexdigest()}
        used=r.get('usage',{}).get('input_tokens')
        if used is None and isinstance(coverage,dict):used=sum(v.get('input_tokens',0) for v in coverage.values() if isinstance(v,dict))
        row['accounted_input_tokens']=used
        emit(out/'rows.jsonl',row);total_tokens+=used or 0;assert total_tokens<=500000
        print(json.dumps({'case':c['id'],'phase':phase,'status':status,'seconds':elapsed}),flush=True)
        if status=='FAIL':raise RuntimeError('recorded failure; no silent request retry')
    (out/'completion.json').write_text(json.dumps({'status':'PASS','requests':len(plan),'decision_sequences':sequences,'reported_input_tokens':total_tokens,'actual_forward_calls':a.forward_calls,'cases_sha256':hashlib.sha256(raw).hexdigest(),'failure_count':failures},indent=2))
if __name__=='__main__':main()
