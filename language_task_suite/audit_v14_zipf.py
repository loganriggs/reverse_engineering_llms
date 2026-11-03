"""Matched word-rank diagnostics for V14 and a frozen FineWeb panel."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER
from audit_surface_repetition import WORD
from surface_mixture_v6 import render
from surface_mixture_v14_conditional import SurfaceMixtureV14Conditional
from train_v13_attributes import COMMON, REFERENCE


def words(text,n):
    return [w.lower() for w in WORD.findall(text)[:n]]


def summarize(rows,n):
    counts=Counter(word for row in rows for word in row[:n])
    rank=np.asarray(sorted(counts.values(),reverse=True),dtype=float)
    total=int(rank.sum())
    lo,hi=10,min(100,len(rank))
    slope=float(np.polyfit(np.log(np.arange(lo,hi+1)),
                           np.log(rank[lo-1:hi]/total),1)[0]) if hi>lo else None
    return counts,rank,dict(n_windows=len(rows),words_per_window=n,
        total_words=total,observed_vocabulary=len(counts),
        top1_mass=float(rank[0]/total),top10_mass=float(rank[:10].sum()/total),
        tail_slope_rank10_to100=slope,
        hapax_types=sum(v==1 for v in counts.values()),
        top25=counts.most_common(25))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rows',type=Path,default=DEFAULT_ROWS)
    p.add_argument('--tokenizer',type=Path,default=DEFAULT_TOKENIZER)
    p.add_argument('--n',type=int,default=480)
    p.add_argument('--words',type=int,default=100)
    p.add_argument('--toy-length',type=int,default=256)
    p.add_argument('--seed',type=int,default=99851)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    frozen=torch.load(args.rows,map_location='cpu',weights_only=True)
    assert args.n<=len(frozen)
    tokenizer=Tokenizer.from_file(str(args.tokenizer))
    natural=[words(tokenizer.decode(row.tolist(),skip_special_tokens=False),
                   args.words) for row in frozen[:args.n]]
    gen=SurfaceMixtureV14Conditional(seed=args.seed,**COMMON,**REFERENCE,
                                     conditional_continuation_rate=.35)
    toy=[]
    for _ in range(args.n):
        ids,_,_=gen.sample_sequence(args.toy_length)
        toy.append(words(render([gen.id2token[i] for i in ids]),args.words))
    eligible=[i for i in range(args.n)
              if len(natural[i])>=args.words and len(toy[i])>=args.words]
    assert len(eligible)>=args.n//2
    counts={};ranks={};summary={}
    for name,rows in [('FineWeb',natural),('V14',toy)]:
        counts[name],ranks[name],summary[name]=summarize(
            [rows[i] for i in eligible],args.words)
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    for name,color in [('FineWeb','#3569a4'),('V14','#c46b36')]:
        rank=ranks[name];ax.loglog(np.arange(1,len(rank)+1),rank/rank.sum(),
                                  marker='.',ms=2,lw=1,color=color,label=name)
    ax.set(xlabel='Word frequency rank',ylabel='Fraction of matched words',
           title='Word-rank curve on matched 100-word windows')
    ax.legend()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output_dir/'v14_vs_fineweb_word_rank.png',dpi=180)
    plt.close(fig)
    result=dict(config={**vars(args),'rows':str(args.rows),
                        'tokenizer':str(args.tokenizer),
                        'output_dir':str(args.output_dir)},
                n_eligible=len(eligible),summary=summary,
                ranks={key:value.tolist() for key,value in ranks.items()},
                caveat='Frozen local FineWeb token windows, not independent '
                       'documents; word regex differs from model tokens. '
                       'Rank slope is a descriptive fit over ranks 10-100, '
                       'not a test of Zipfian generative equivalence. '
                       'Frequency match alone does not imply syntactic or '
                       'semantic realism.')
    (args.output_dir/'v14_vs_fineweb_word_rank.json').write_text(
        json.dumps(result,indent=2)+'\n')
    for name,value in summary.items():print(name,value,flush=True)


if __name__=='__main__':main()
