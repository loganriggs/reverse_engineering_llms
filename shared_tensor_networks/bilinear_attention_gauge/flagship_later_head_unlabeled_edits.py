"""Balanced answer-free token edits for later-head exact gauge fitting."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_rank_pilot import HeadCut,final_logits
from flagship_matrix_free import CORPUS,SNAPSHOT
from flagship_transfer import ART

N_PER_TYPE=80
TYPES=('bracket','noun_number','adjective','color','country','name',
       'number','punctuation','preposition','random_context')
SEEDS={'train':20261111,'validation':20261121}


def draw_pair(kind,rng):
    words=lambda v: str(rng.choice(v))
    if kind=='bracket':
        prefix=words(('Try','Evaluate','Compute','Consider','Check'))
        op=words(('*','/','-','and',','))
        a,b=rng.integers(1,10,size=2)
        return f'{prefix} ({a} {op} {b}',f'{prefix} [{a} {op} {b}'
    if kind=='noun_number':
        noun=words(('bird','doctor','apple','table','river','teacher','flower','cat','book','car'))
        irregular={'bird':'birds','doctor':'doctors','apple':'apples','table':'tables',
                   'river':'rivers','teacher':'teachers','flower':'flowers','cat':'cats',
                   'book':'books','car':'cars'}
        prefix=words(('Look at','Imagine','I noticed','We observed','Think of'))
        prep=words(('beside','behind','near','above','below'))
        distract=words(('trees','rooms','people','houses','windows'))
        return (f'{prefix} the {noun} {prep} the {distract}',
                f'{prefix} the {irregular[noun]} {prep} the {distract}')
    if kind=='adjective':
        a,b=[str(v) for v in rng.choice(('hot','cold','fast','slow','bright','dark',
                                         'wet','dry','large','small'),size=2,replace=False)]
        noun=words(('road','house','day','water','car','room'))
        suffix=words(('was surprising','seemed strange','looked different','is here'))
        return f'The {a} {noun} {suffix}',f'The {b} {noun} {suffix}'
    if kind=='color':
        a,b=[str(v) for v in rng.choice(('red','blue','green','yellow','black','white'),size=2,replace=False)]
        noun=words(('box','lamp','door','shirt','car','book'))
        suffix=words(('is nearby','contains a note','was found','looks new'))
        return f'The {a} {noun} {suffix}',f'The {b} {noun} {suffix}'
    if kind=='country':
        a,b=[str(v) for v in rng.choice(('France','Germany','Italy','Japan','Spain','China',
                                         'Canada','Egypt','Russia','Greece','India','Portugal'),size=2,replace=False)]
        prefix=words(('Visitors from','A traveler from','Someone in','News from'))
        suffix=words(('arrived today','made the announcement','was mentioned','came yesterday'))
        return f'{prefix} {a} {suffix}',f'{prefix} {b} {suffix}'
    if kind=='name':
        a,b=[str(v) for v in rng.choice(('Alice','Bob','Carol','David','Emma','Frank',
                                         'Grace','Henry','Irene','Jack','Liam','Noah',
                                         'Olivia','Sophia','Mia','Ethan','Ava','Lucas',
                                         'Chloe','Mason'),size=2,replace=False)]
        suffix=words(('said the answer was','wrote a letter','entered the room',
                      'spoke to the team','found the book','opened the window',
                      'visited the city','called the teacher'))
        prefix=words(('Then','Later','Meanwhile','Yesterday','Today'))
        return f'{prefix} {a} {suffix}',f'{prefix} {b} {suffix}'
    if kind=='number':
        a,b=rng.choice(np.arange(1,20),size=2,replace=False)
        prefix=words(('After','Before','At position','Consider number','We counted'))
        suffix=words(('comes a surprise','in the list','is the value','was written'))
        return f'{prefix} {a} {suffix}',f'{prefix} {b} {suffix}'
    if kind=='punctuation':
        a,b=[str(v) for v in rng.choice((',', ';', ':', '.'),size=2,replace=False)]
        subject=words(('Alice','Bob','The teacher','The driver','A friend',
                       'The scientist','A neighbor','The student','Her friend','Our guide'))
        suffix=words(('the story continued','we listened','the room was quiet',
                      'everyone waited','the result appeared','nobody spoke',
                      'the lights changed','people laughed','the door opened',
                      'the answer arrived'))
        return f'{subject} said {a} {suffix}',f'{subject} said {b} {suffix}'
    if kind=='preposition':
        a,b=[str(v) for v in rng.choice(('above','below','behind','beside','near'),size=2,replace=False)]
        noun=words(('cat','book','lamp','box','car'))
        other=words(('table','window','door','tree','house'))
        suffix=words(('was visible','looked different','remained there','seemed odd'))
        return f'The {noun} {a} the {other} {suffix}',f'The {noun} {b} the {other} {suffix}'
    raise ValueError(kind)


def generate(tokenizer,panel,excluded=frozenset()):
    rng=np.random.default_rng(SEEDS[panel])
    corpus=np.load(CORPUS,mmap_mode='r')
    docs=(0,16000) if panel=='train' else (17500,20000)
    cases=[];attempts={}
    encoded_pool=[tokenizer.encode(x,add_special_tokens=False).ids
                  for x in (' red',' blue',' green',' black',' dog',' cat',' house',' book',
                            ' Paris',' Berlin',' one',' two',' three',' Alice',' Bob')]
    assert all(len(ids)==1 for ids in encoded_pool)
    pool=[ids[0] for ids in encoded_pool]
    for kind in TYPES:
        seen=set();count=0;tries=0
        while count<N_PER_TYPE:
            tries+=1
            assert tries<30000,(kind,count)
            if kind=='random_context':
                doc=int(rng.integers(*docs));start=int(rng.integers(0,corpus.shape[1]-16+1))
                before=np.asarray(corpus[doc,start:start+16],dtype=np.int64).tolist()
                after=before.copy();pos=int(rng.integers(0,15))
                after[pos]=int(rng.choice(pool))
                text_a=text_b=None
            else:
                text_a,text_b=draw_pair(kind,rng)
                before=tokenizer.encode(text_a,add_special_tokens=False).ids
                after=tokenizer.encode(text_b,add_special_tokens=False).ids
            if len(before)!=len(after) or sum(a!=b for a,b in zip(before,after))!=1:
                continue
            if rng.random()<.5:
                before,after=after,before
                text_a,text_b=text_b,text_a
            key=tuple(sorted((tuple(before),tuple(after))))
            if key in seen or key in excluded:continue
            seen.add(key);count+=1
            cases.append(dict(type=kind,base_ids=before,edited_ids=after,
                              base_text=text_a,edited_text=text_b))
        attempts[kind]=tries
    rng.shuffle(cases)
    return cases,attempts


def capture(model,probes,bases,cases):
    out={t:np.zeros((len(cases),4),dtype=np.float32) for t in TARGETS}
    by_length={}
    for i,case in enumerate(cases):by_length.setdefault(len(case['base_ids']),[]).append(i)
    with torch.no_grad():
        for length,indices in sorted(by_length.items()):
            for start in range(0,len(indices),16):
                ix=indices[start:start+16]
                before=torch.tensor([cases[i]['base_ids'] for i in ix],dtype=torch.long)
                after=torch.tensor([cases[i]['edited_ids'] for i in ix],dtype=torch.long)
                final_logits(model,before,stop_at=max(probes))
                a={t:probes[layer].last_input[:,-1,head*probes[layer].head_dim:(head+1)*probes[layer].head_dim].clone()
                   for t,(layer,head) in TARGETS.items()}
                final_logits(model,after,stop_at=max(probes))
                for t,(layer,head) in TARGETS.items():
                    p=probes[layer];sl=slice(head*p.head_dim,(head+1)*p.head_dim)
                    delta=p.last_input[:,-1,sl]-a[t]
                    out[t][ix]=(delta@bases[t]).numpy()
    return out


def run():
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    panels={};attempts={}
    panels['train'],attempts['train']=generate(tokenizer,'train')
    excluded={tuple(sorted((tuple(c['base_ids']),tuple(c['edited_ids'])))) for c in panels['train']}
    panels['validation'],attempts['validation']=generate(tokenizer,'validation',excluded)
    assert not excluded.intersection(
        tuple(sorted((tuple(c['base_ids']),tuple(c['edited_ids']))))
        for c in panels['validation'])
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probes={layer:HeadCut(model.transformer.h[layer].attn)
            for layer in sorted(set(layer for layer,_ in TARGETS.values()))}
    saved=np.load(ART/'later_head_rank_pilot_bases.npz')
    bases={t:torch.tensor(saved[f'{t}__weight'][:,:4].copy(),dtype=torch.float32) for t in TARGETS}
    arrays={}
    for name,cases in panels.items():
        result=capture(model,probes,bases,cases)
        for t,x in result.items():arrays[f'{t}__{name}']=x
        print(name,len(cases),{k:attempts[name][k] for k in TYPES},flush=True)
    np.savez(ART/'later_head_unlabeled_edits.npz',**arrays)
    (ART/'later_head_unlabeled_edits.json').write_text(json.dumps(dict(
        description='Balanced answer-free token substitutions across ten edit types; two independent seeds and disjoint random-context docs',
        n_per_type=N_PER_TYPE,seeds=SEEDS,attempts=attempts,panels=panels),indent=2)+'\n')


if __name__=='__main__':run()
