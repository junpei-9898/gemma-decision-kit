# SPDX-License-Identifier: Apache-2.0
"""Frozen synthetic paired probes. No model-generated gold or input truncation.

Use build_body(tokenizer, case, target_tokens) with the pinned model tokenizer.
Target 0 produces an unpadded short control. Other targets include chat framing.
This is a small controlled probe, not a natural-document accuracy benchmark.
"""
import random
from functools import lru_cache
from gemma_decision.core import prompt_text

CRITERIA = {'supported':'記録が主張を支持する。',
            'refuted':'記録が主張に明確に反する。',
            'insufficient':'記録だけでは判断できない。'}

@lru_cache(maxsize=1)
def background_records():
    rng=random.Random(20260924)
    places=['北倉庫','南倉庫','資料室','作業室','搬入口','休憩室']
    objects=['棚','照明','机','台車','換気設備','掲示板','清掃用具']
    activities=['点検を実施した','配置を確認した','清掃を完了した','数量を確認した','記録用の写真を撮影した']
    records=[]
    for i in range(20000):
        records.append(f'業務記録{i+1:05d}：{rng.choice(places)}の{rng.choice(objects)}について{rng.choice(activities)}。確認した項目は{rng.randint(2,38)}件で、作業は{rng.randint(8,16)}時{rng.randrange(0,60,5):02d}分に終了した。\n')
    return records


def prompt_ids(tokenizer,body):
    ids=tokenizer.apply_chat_template([{'role':'user','content':prompt_text(body['state'],body['questions']['q'])}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    return ids if isinstance(ids,list) else ids['input_ids']


def build_body(tokenizer,case,target_tokens):
    records=background_records()
    question={'type':'choice','instructions':'記録だけを根拠に判定してください。対象主張：'+case['claim']+'一般知識で補わないでください。','criteria':CRITERIA}
    def body(n):
        split={'begin':0,'middle':n//2,'end':n}[case['position']]
        state=''.join(records[:split])+case['evidence']+'\n'+''.join(records[split:n])
        return {'state':state,'questions':{'q':question}}
    if target_tokens==0:
        result=body(0)
        return result,prompt_ids(tokenizer,result)
    lo,hi=0,len(records)
    while lo<hi:
        mid=(lo+hi+1)//2
        if len(prompt_ids(tokenizer,body(mid)))<=target_tokens:lo=mid
        else:hi=mid-1
    result=body(lo);ids=prompt_ids(tokenizer,result)
    assert target_tokens-128<=len(ids)<=target_tokens
    assert len(result['state'])<=2000000
    return result,ids
