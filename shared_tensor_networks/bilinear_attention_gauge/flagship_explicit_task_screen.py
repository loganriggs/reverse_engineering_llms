"""Screen explicit next-token behaviors before any 546M gauge fitting."""

import argparse
import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_full_model_eval import GPT, GPTConfig, logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


TOKENIZER='/home/loganriggs/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json'


def examples(seed,n=20):
    rng=np.random.default_rng(seed)
    out=[]
    names=['Alice','Bob','Carol','David','Emma','Frank']
    colors=[' red',' blue',' green',' yellow',' black',' white']
    for _ in range(n):
        a,b=rng.choice(names,size=2,replace=False)
        x,y=rng.choice(colors,size=2,replace=False)
        query=int(rng.integers(0,2))
        out.append(('association_retrieval',
            f'{a} owns the{x} key. {b} owns the{y} key. '+
            f'{a if query==0 else b} owns the',x if query==0 else y,y if query==0 else x))
    nouns=[('key','keys','cabinet','cabinets'),('dog','dogs','cat','cats'),
           ('book','books','shelf','shelves'),('car','cars','garage','garages'),
           ('student','students','teacher','teachers')]
    for _ in range(n):
        singular,plural,dist_s,dist_p=nouns[int(rng.integers(len(nouns)))]
        is_plural=bool(rng.integers(0,2))
        subj=plural if is_plural else singular
        distract=dist_s if is_plural else dist_p
        out.append(('subject_verb_agreement',f'The {subj} near the {distract}',
                    ' are' if is_plural else ' is',' is' if is_plural else ' are'))
    for _ in range(n):
        a=int(rng.integers(0,7))
        out.append(('number_sequence',f'{a}, {a+1}, {a+2},',
                    f' {a+3}',f' {a+4}'))
    for _ in range(n):
        a=int(rng.integers(1,10));b=int(rng.integers(1,10))
        out.append(('addition',f'{a} + {b} =',f' {a+b}',f' {a+b+1}'))
    capitals=[('France',' Paris'),('Germany',' Berlin'),('Italy',' Rome'),
              ('Japan',' Tokyo'),('Spain',' Madrid'),('China',' Beijing'),
              ('Canada',' Ottawa'),('Egypt',' Cairo'),('Russia',' Moscow'),
              ('Greece',' Athens'),('India',' Delhi'),('Portugal',' Lisbon')]
    for i in rng.permutation(len(capitals))[:min(n,len(capitals))]:
        country,correct=capitals[int(i)]
        foil=capitals[int(rng.choice([j for j in range(len(capitals)) if j!=i]))][1]
        out.append(('country_capital',f'The capital of {country} is',correct,foil))
    opposites=[('hot',' cold',' warm'),('big',' small',' large'),
               ('happy',' sad',' joyful'),('dark',' light',' black'),
               ('fast',' slow',' quick'),('old',' young',' ancient'),
               ('high',' low',' tall'),('wet',' dry',' damp')]
    for _ in range(n):
        word,correct,foil=opposites[int(rng.integers(len(opposites)))]
        out.append(('antonym',f'The opposite of {word} is',correct,foil))
    for _ in range(n):
        a=int(rng.integers(1,10));b=int(rng.integers(1,10))
        opening='(' if rng.random()<.5 else '['
        out.append(('bracket_closure',f'{opening}{a} + {b}',
                    ')' if opening=='(' else ']',']' if opening=='(' else ')'))
    rng.shuffle(out)
    return out


def run(seed=9611,n=20):
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    generated=examples(seed,n)
    encoded=[]
    for family,prompt,correct,foil in generated:
        c=tokenizer.encode(correct,add_special_tokens=False).ids
        f=tokenizer.encode(foil,add_special_tokens=False).ids
        assert len(c)==len(f)==1,(family,correct,foil,c,f)
        encoded.append(dict(family=family,prompt=prompt,correct=correct,
                            foil=foil,correct_id=c[0],foil_id=f[0],
                            prompt_ids=tokenizer.encode(prompt,add_special_tokens=False).ids))
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    with torch.no_grad():
        for i,row in enumerate(encoded):
            value=logits(model,torch.tensor(row['prompt_ids'])[None,:])[0,-1].float()
            margin=float(value[row['correct_id']]-value[row['foil_id']])
            row['margin']=margin
            row['forced_choice_correct']=margin>0
            row['correct_token_top1']=int(value.argmax())==row['correct_id']
            if (i+1)%20==0:print(i+1,len(encoded),flush=True)
    summary={}
    for family in sorted(set(row['family'] for row in encoded)):
        subset=[row for row in encoded if row['family']==family]
        summary[family]={'n':len(subset),
            'forced_choice_accuracy':float(np.mean([row['forced_choice_correct'] for row in subset])),
            'correct_token_top1_rate':float(np.mean([row['correct_token_top1'] for row in subset])),
            'mean_margin':float(np.mean([row['margin'] for row in subset]))}
    result={'seed':seed,'n_requested_per_family':n,'summary':summary,'examples':encoded,
            'scope':'native 546M next-token logits, single-token forced-choice; no gauge fitting'}
    path=ART/f'all_head_explicit_task_screen_s{seed}_n{n}.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(summary,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=9611)
    parser.add_argument('--n',type=int,default=20)
    args=parser.parse_args();run(args.seed,args.n)
