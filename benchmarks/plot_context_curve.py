# SPDX-License-Identifier: Apache-2.0
"""Rebuild documentation charts: python benchmarks/plot_context_curve.py
Requires matplotlib only for plotting; not a runtime dependency of the kit.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FixedFormatter

ROOT=Path(__file__).resolve().parents[1]
def main():
    data=json.loads((ROOT/'docs/context-accuracy-results.json').read_text())
    pairs=data['paired_curve'];stress=data['historical_stress']
    assert all(p['completed_cases']==p['expected_cases'] and p['accuracy'] is not None for p in pairs)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,(a,b)=plt.subplots(2,1,figsize=(12,9),sharex=True,gridspec_kw={'height_ratios':[1.3,1]})
    fig.patch.set_facecolor('#f8fafc')
    fig.suptitle('Longer context: measured accuracy and processing time',fontsize=20,fontweight='bold',x=.075,ha='left',y=.97)
    fig.text(.075,.921,'Gemma Decision Kit · NVFP4 · NVIDIA GB10 · full input, no prefix reuse',fontsize=12,color='#475569')
    x=[v['median_prompt_tokens'] for v in pairs];y=[100*v['accuracy'] for v in pairs]
    a.plot(x,y,'o-',color='#1d4ed8',lw=2.5,ms=7,label='Varied-background paired probes (9 cases / length)')
    a.plot([v['median_prompt_tokens'] for v in stress],[100*v['accuracy'] for v in stress],'s--',color='#c2410c',lw=2,ms=6,label='Earlier repeated-negative stress probes (7 cases / length)')
    for xx,yy,v in zip(x,y,pairs):
        a.annotate(f"{v['correct']}/{v['expected_cases']}",(xx,yy),xytext=(0,10 if yy<85 else -19),textcoords='offset points',ha='center',color='#1d4ed8',fontweight='bold')
    a.set_ylim(-3,106);a.set_ylabel('Agreement with provisional labels (%)');a.legend(loc='lower left',frameon=False,fontsize=10)
    b.plot(x[:-1],[v['median_seconds'] for v in pairs[:-1]],'o-',color='#047857',lw=2.5,ms=7)
    b.scatter(x[-1],pairs[-1]['median_seconds'],marker='D',s=65,color='#7c3aed',label='256K: resumed process, reference timing')
    b.legend(loc='upper left',frameon=False,fontsize=9)
    for xx,v in zip(x,pairs):
        b.annotate(f"{v['median_seconds']:.2f}s",(xx,v['median_seconds']),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10,color='#047857')
    b.set_yscale('log');b.set_ylabel('Median HTTP time (s, log scale)');b.set_xlabel('Actual prompt tokens (log scale; short control has no filler)')
    b.set_ylim(min(v['median_seconds'] for v in pairs)/1.6,max(v['median_seconds'] for v in pairs)*2.3)
    ticks=[x[0],8192,32768,65536,131072,262144]
    for ax in [a,b]:
        ax.set_xscale('log',base=2);ax.set_facecolor('#ffffff');ax.grid(True,alpha=.18);ax.set_xlim(90,340000)
        ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(FixedFormatter(['Short','8K','32K','64K','128K','256K']))
    fig.text(.075,.044,'Exploratory synthetic tests with frozen AI-provisional labels; not a general accuracy benchmark.\nFixed 256K context / 8GiB KV; one run per case/length. 256K resumed with concurrency authorized.\nOrange: historical stress set. Timings include incorrect answers; purple point is not an isolated speed comparison.',fontsize=10,color='#475569',va='bottom')
    fig.subplots_adjust(left=.09,right=.96,top=.88,bottom=.16,hspace=.13)
    output=ROOT/'docs/assets';output.mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(output/f'context-accuracy.{ext}',dpi=160,facecolor=fig.get_facecolor())
    svg=output/'context-accuracy.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
if __name__=='__main__':main()
