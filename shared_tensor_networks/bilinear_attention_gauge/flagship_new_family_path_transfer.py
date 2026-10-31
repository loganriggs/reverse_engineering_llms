"""Whole-edit-family transfer for L9H7 rank-32 reduction measures."""

import json
import os

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_all_head_rank_frontier import top_basis
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT, GPTConfig
from flagship_later_head_rank_pilot import HeadCut, final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

LAYER,HEAD,RANK=9,7,int(os.environ.get('ATTN_RANK','32'))
SEED=20261031
N_PER_FAMILY=12
FAMILIES=('modal','connective','tense','code_relation')


def draw(kind,rng):
    word=lambda choices:str(rng.choice(choices))
    if kind=='modal':
        subject=word(('The robot','The student','A visitor','The driver','The team'))
        verb=word(('open the door','move the box','finish the task',
                   'read the note','start the engine'))
        ending=word(('today','tomorrow','after lunch','before noon'))
        a,b=rng.choice(('can','must','might','will'),size=2,replace=False)
        return f'{subject} {a} {verb} {ending}',f'{subject} {b} {verb} {ending}'
    if kind=='connective':
        first=word(('the lamp is on','the gate is open','the signal is red',
                    'the worker is here','the test passes'))
        second=word(('the door is open','the machine is ready','the room is quiet',
                     'the file exists','the battery is full'))
        suffix=word(('we proceed','the alarm rings','the system starts',
                     'the light flashes','the result is valid'))
        return f'If {first} and {second}, {suffix}',f'If {first} or {second}, {suffix}'
    if kind=='tense':
        subject=word(('The worker','The teacher','The child','The visitor','The driver'))
        action=word(('reading the document','walking through the room',
                     'writing the report','looking at the clock','waiting by the gate'))
        ending=word(('when we arrived','during the event','at the station',
                     'near the entrance'))
        return f'{subject} was {action} {ending}',f'{subject} is {action} {ending}'
    if kind=='code_relation':
        left=word(('x','a','count','value','index'))
        right=word(('y','b','limit','target','size'))
        suffix=word(('returns true','runs the branch','sets the flag',
                     'ends the loop','prints a line'))
        return (f'In code, {left} == {right} {suffix}',
                f'In code, {left} != {right} {suffix}')
    raise ValueError(kind)


def cases(tokenizer):
    rng=np.random.default_rng(SEED)
    previous=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    excluded={tuple(sorted((tuple(r['base_ids']),tuple(r['edited_ids']))))
              for panel in previous['panels'].values() for r in panel}
    records=[]
    for kind in FAMILIES:
        made=0;attempts=0
        while made<N_PER_FAMILY:
            attempts+=1
            assert attempts<10000,(kind,made)
            a,b=draw(kind,rng)
            x=tokenizer.encode(a,add_special_tokens=False).ids
            y=tokenizer.encode(b,add_special_tokens=False).ids
            if len(x)!=len(y) or sum(u!=v for u,v in zip(x,y))!=1:continue
            if rng.random()<.5:x,y=y,x;a,b=b,a
            key=tuple(sorted((tuple(x),tuple(y))))
            if key in excluded:continue
            excluded.add(key)
            records.append(dict(type=kind,base_ids=x,edited_ids=y,
                                base_text=a,edited_text=b))
            made+=1
        print('generated',kind,made,'in',attempts,'attempts',flush=True)
    return records


def run():
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    panel=cases(tokenizer)
    saved=np.load(ART/'all_head_unlabeled_edit_scan.npz')
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    path=json.loads((ART/'taskblind_head_path_scan.json').read_text())
    guardrail=json.loads((ART/'path_floor_guardrail.json').read_text())
    train=saved[f'layer{LAYER}__train'][:,HEAD].astype(np.float64)
    train_types=np.array([r['type'] for r in meta['panels']['train']])
    names=sorted(set(train_types))
    path_train=next(r for r in path['summary'] if
                    (r['layer'],r['head'],r['panel'])==(LAYER,HEAD,'train'))
    chosen=sorted(names,key=lambda name:
                  -path_train['mean_by_type'][name]['path_l2'])[:3]
    policy=next(r['policy'] for r in guardrail['rows'] if
                (r['layer'],r['head'],r['rank'])==(LAYER,HEAD,RANK))
    assert policy['method'].startswith('floor_')
    floor=float(policy['method'].split('_')[1])
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probe=HeadCut(model.transformer.h[LAYER].attn);probe.head=HEAD
    projection=model.transformer.h[LAYER].attn.c_proj.weight.detach().numpy()
    weight=projection[:,HEAD*128:(HEAD+1)*128].astype(np.float64)
    gram=weight.T@weight
    def normalized(name):
        x=train[train_types==name];cov=x.T@x/len(x)
        return cov/max(np.trace(cov@gram),1e-30)
    balanced=sum(normalized(name) for name in names)/len(names)
    target=sum(normalized(name) for name in chosen)/len(chosen)
    moments={'weight_only':gram,'pooled_edit':train.T@train/len(train),
             'fixed_floor_0.1':.9*target+.1*balanced,
             'training_guardrail_floor':(1-floor)*target+floor*balanced}
    bases={name:torch.tensor(top_basis(cov/np.trace(cov))[:,:RANK].copy(),
                              dtype=torch.float32) for name,cov in moments.items()}
    rows=[]
    with torch.no_grad():
        for index,case in enumerate(panel):
            before=torch.tensor(case['base_ids'],dtype=torch.long)[None,:]
            after=torch.tensor(case['edited_ids'],dtype=torch.long)[None,:]
            probe.basis=None;probe.injection=None
            final_logits(model,before,stop_at=LAYER)
            base_state=probe.last_input[0,-1,HEAD*128:(HEAD+1)*128].clone()
            final_logits(model,after,stop_at=LAYER)
            delta=probe.last_input[0,-1,HEAD*128:(HEAD+1)*128]-base_state
            probe.basis=torch.eye(128)
            native_base=final_logits(model,before)[0].double()
            probe.injection=delta
            native_patched=final_logits(model,before)[0].double()
            probe.injection=None
            native_effect=native_patched-native_base
            denominator=float(native_effect.square().sum())
            assert denominator>1e-12,(case['type'],index)
            for name,basis in bases.items():
                probe.basis=basis
                cut_base=final_logits(model,before)[0].double()
                p=torch.log_softmax(native_base,dim=-1)
                q=torch.log_softmax(cut_base,dim=-1)
                probe.injection=(delta@basis)@basis.T
                cut_patched=final_logits(model,before)[0].double()
                probe.injection=None
                effect=cut_patched-cut_base
                rows.append(dict(index=index,type=case['type'],method=name,
                    relative_path_mse=float((effect-native_effect).square().sum()/denominator),
                    path_cosine=float((effect@native_effect)/
                                      (effect.norm()*native_effect.norm()).clamp_min(1e-30)),
                    forward_base_kl=float((p.exp()*(p-q)).sum()),
                    native_path_l2=float(native_effect.norm())))
            probe.basis=None
            print('case',index,case['type'],flush=True)
    result=dict(description='Whole-family out-of-distribution edit test. Four newly designed edit families absent from all fitting, path selection, and floor selection. Native edit state captured from full model; head-patch final-logit paths compared after each reduced cut. Ordinary base-prompt forward KL also recorded. No task answer labels used. Tested families are designed, so this is not blind discovery.',
                layer=LAYER,head=HEAD,rank=RANK,seed=SEED,
                n_per_family=N_PER_FAMILY,training_selected_types=chosen,
                training_guardrail_floor=floor,cases=panel,rows=rows)
    (ART/f'new_family_path_transfer_rank{RANK}.json').write_text(
        json.dumps(result,indent=2)+'\n')
    for kind in FAMILIES:
        for name in moments:
            part=[r for r in rows if r['type']==kind and r['method']==name]
            print(kind,name,'path mse',round(float(np.mean([
                r['relative_path_mse'] for r in part])),4),'base KL',round(float(
                np.mean([r['forward_base_kl'] for r in part])),5),flush=True)


if __name__=='__main__':run()
