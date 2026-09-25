# SPDX-License-Identifier: Apache-2.0
# @unit evaluation
# @layer infrastructure
# @work-item-id WI-063
"""Explicit three-valued quantifiers; never guess semantics from arbitrary prose."""
from ..eider_engine import validate


def split_request(value, validator=validate):
    if not isinstance(value,dict):raise ValueError('Invalid unified request')
    body={k:v for k,v in value.items() if k!='aggregation'}
    validator(body)
    policies=value.get('aggregation',{})
    if not isinstance(policies,dict) or any(q not in body['questions'] for q in policies):raise ValueError('Invalid aggregation question')
    for key,p in policies.items():
        if not isinstance(p,dict) or set(p)!={'operator','positive','negative','unknown'}:raise ValueError('Invalid aggregation policy')
        if p['operator'] not in ('any','all'):raise ValueError('Aggregation operator must be any/all')
        if body['questions'][key]['type']!='choice':raise ValueError('Aggregation supports choice only')
        labels=[p[k] for k in ('positive','negative','unknown')]
        if any(not isinstance(x,str) for x in labels) or set(labels)!=set(body['questions'][key]['criteria']):raise ValueError('Aggregation must map all three distinct criteria')
    return body,policies


def reduce_windows(evidence,key,policy):
    values=[(e['id'],e['decision']['answers'][key]['choice']) for e in evidence]
    positive,negative,unknown=(policy[k] for k in ('positive','negative','unknown'))
    dominant=positive if policy['operator']=='any' else negative
    fallback=negative if policy['operator']=='any' else positive
    if any(v==dominant for _,v in values):choice=dominant
    elif all(v==fallback for _,v in values):choice=fallback
    else:choice=unknown
    return {'type':'choice','choice':choice,'probabilities':None,'aggregation':policy['operator'],
            'evidence_windows':[wid for wid,v in values if v==choice],
            'scope':'logical reduction of provisional sampled-window judgments'}
