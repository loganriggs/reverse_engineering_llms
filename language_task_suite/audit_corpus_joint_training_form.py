"""Compare actual joint-training row allocation with unselected prose form."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_document_structure_vs_fineweb import structure
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from audit_surface_repetition import WORD
from compare_v12_fineweb_fresh import first_words


def _text(row,index,tokenizer,words):
    x,y=row['x'][index],row['y'][index]
    ids=x[y!=-100].tolist()
    text=first_words(tokenizer.decode(ids,skip_special_tokens=True),words)
    return text if len(WORD.findall(text))>=words else None


def _sample_rows(row,n,tokenizer,rng,words):
    indices=list(range(len(row['x'])))
    rng.shuffle(indices)
    out=[]
    for i in indices:
        text=_text(row,i,tokenizer,words)
        if text is not None:
            out.append(text)
            if len(out)==n:
                break
    if len(out)<n:
        raise ValueError(f'Only {len(out)} usable rows of {n}')
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--fresh-preview',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,default=512)
    p.add_argument('--words',type=int,default=100)
    p.add_argument('--seed',type=int,default=46501)
    a=p.parse_args()
    if a.n%8:
        raise ValueError('Panel size must be divisible by eight')
    cache=torch.load(a.cache,map_location='cpu',weights_only=True)
    tokenizer=Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng=random.Random(a.seed)
    train=cache['train']
    background=_sample_rows(train['background'],a.n,tokenizer,rng,a.words)
    mixed_bg=_sample_rows(train['background'],a.n//4,tokenizer,rng,a.words)
    mixed_bracket=_sample_rows(train['sites']['bracket'],a.n//4,
                               tokenizer,rng,a.words)
    mixed_code=_sample_rows(train['sites']['literal_copy'],a.n//8,
                            tokenizer,rng,a.words)
    mixed_style=_sample_rows(train['sites']['natural_style'],a.n//8,
                             tokenizer,rng,a.words)
    left,right=train['name_pairs']
    eligible=[i for i in range(len(left['x']))
              if _text(left,i,tokenizer,a.words) is not None and
              _text(right,i,tokenizer,a.words) is not None]
    rng.shuffle(eligible)
    if len(eligible)<a.n//8:
        raise ValueError('Too few eligible paired name rows')
    paired=[]
    for i in eligible[:a.n//8]:
        paired.extend((_text(left,i,tokenizer,a.words),
                       _text(right,i,tokenizer,a.words)))
    actual=mixed_bg+paired+mixed_bracket+mixed_code+mixed_style
    # Same family mix without two near-identical counterfactual arms. This
    # isolates duplicate-pair contribution to cross-document n-gram reuse.
    if len(eligible)<a.n//4:
        raise ValueError('Too few distinct name pairs for unpaired panel')
    one_per_pair=[_text(left,i,tokenizer,a.words)
                  for i in eligible[:a.n//4]]
    unpaired=mixed_bg+one_per_pair+mixed_bracket+mixed_code+mixed_style
    one_arm_background=_sample_rows(train['background'],3*a.n//8,
                                     tokenizer,rng,a.words)
    causal_only_mix=(one_arm_background+one_per_pair[:a.n//8]+
                     mixed_bracket+mixed_code+mixed_style)
    fresh_rows=json.loads(a.fresh_preview.read_text())
    fresh=[]
    for row in fresh_rows:
        text=first_words(row['preview'],a.words)
        if len(WORD.findall(text))>=a.words:
            fresh.append(text)
        if len(fresh)==a.n:
            break
    if len(fresh)<a.n:
        raise ValueError('Too few fresh reference texts')
    assert (len(actual)==len(unpaired)==len(causal_only_mix)==
            len(background)==len(fresh)==a.n)
    panels={'fresh_fineweb':structure(fresh),
            'unselected_training_background':structure(background),
            'actual_eight_row_allocation':structure(actual),
            'same_family_mix_one_name_arm_per_source':structure(unpaired),
            'causal_only_counterfactual_allocation':structure(causal_only_mix)}
    report={'source':'HuggingFaceFW/fineweb sample-10BT train',
            'config':{'n':a.n,'words':a.words,'seed':a.seed,
                      'train_allocation':{'background':2,'surname_arms':2,
                                          'bracket':2,'code':1,'style':1}},
            'fresh_offsets':[fresh_rows[0]['index'],
                             fresh_rows[-1]['index']],
            'panels':panels,
            'scope':'Descriptive row-form comparison. Training allocates '
                    '75% of rows to selected sites; the paired surname '
                    'arms can raise cross-document n-gram duplication. '
                    'The causal-only allocation uses 3 background, 1 name, '
                    '2 bracket, 1 code, and 1 style LM rows, with a separate '
                    'counterfactual forward used only at the name site. '
                    'Each panel contains the first 100 words of a chosen '
                    '256-token chunk except fresh FineWeb, which uses the '
                    'first 100 words of source previews. Chunk-position '
                    'difference limits direct inference. No semantic '
                    'realism or task-rate equivalence claim. Raw text and '
                    'reconstructable token rows stay local.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    for name,panel in panels.items():
        print(name,'types',panel['word_types'],'top10',round(
            panel['top10_word_share'],3),'crossdoc trigram',round(
            panel['repeated_across_docs_trigram_occurrence_share'],3),
            flush=True)


if __name__=='__main__':
    main()
