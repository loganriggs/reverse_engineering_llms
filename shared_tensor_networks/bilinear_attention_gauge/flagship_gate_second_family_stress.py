"""Second whole-family stress panel for the frozen L9H7 finite-path gate."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_all_head_rank_frontier import top_basis
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT, GPTConfig
from flagship_later_head_rank_pilot import HeadCut, final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

FAMILIES=('quantifier','pronoun','arithmetic_operator','temporal_order',
          'polarity','conditional_operator')
N_PER_FAMILY=12
SEED=20261101


def draw(kind,rng):
    word=lambda choices:str(rng.choice(choices))
    if kind=='quantifier':
        noun=word(('students','workers','visitors','drivers','teachers'))
        action=word(('finished the test','entered the room','opened the gate',
                     'read the notice','received the message'))
        return f'All {noun} {action}',f'Some {noun} {action}'
    if kind=='pronoun':
        verb=word(('opened','found','carried','moved','noticed'))
        obj=word(('the old door','the blue bag','a small note',
                  'the wooden box','a new book'))
        return f'He {verb} {obj}',f'She {verb} {obj}'
    if kind=='arithmetic_operator':
        a,b=rng.integers(1,10,size=2)
        prefix=word(('Calculate','Evaluate','Compute','Check','Solve'))
        suffix=word(('now','carefully','for me','in your head'))
        return f'{prefix} {a} + {b} {suffix}',f'{prefix} {a} * {b} {suffix}'
    if kind=='temporal_order':
        a=word(('the signal','the visitor','the train','the warning','the call'))
        b=word(('the alarm','the meeting','the storm','the event','the announcement'))
        verb=word(('came','arrived','happened','appeared'))
        return f'{a} {verb} before {b}',f'{a} {verb} after {b}'
    if kind=='polarity':
        subject=word(('The answer','The response','The note','The result','The sign'))
        suffix=word(('today','in the report','on the page','for this case'))
        return f'{subject} is yes {suffix}',f'{subject} is no {suffix}'
    if kind=='conditional_operator':
        event=word(('the gate opens','the clock rings','the test passes',
                    'the signal changes','the door closes'))
        action=word(('proceed','stop the machine','send the message',
                     'begin the process','wait outside'))
        return f'If {event}, {action}',f'Unless {event}, {action}'
    raise ValueError(kind)


def generate(tokenizer):
    rng=np.random.default_rng(SEED)
    old=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    first=json.loads((ART/'new_family_path_transfer_rank96.json').read_text())
    previous=[r for panel in old['panels'].values() for r in panel]+first['cases']
    used={tuple(sorted((tuple(r['base_ids']),tuple(r['edited_ids'])))) for r in previous}
    panel=[]
    for kind in FAMILIES:
        tries=0;made=0
        while made<N_PER_FAMILY:
            tries+=1;assert tries<10000,(kind,made)
            a,b=draw(kind,rng)
            x=tokenizer.encode(a,add_special_tokens=False).ids
            y=tokenizer.encode(b,add_special_tokens=False).ids
            if len(x)!=len(y) or sum(i!=j for i,j in zip(x,y))!=1:continue
            if rng.random()<.5:a,b=b,a;x,y=y,x
            key=tuple(sorted((tuple(x),tuple(y))))
            if key in used:continue
            used.add(key)
            panel.append(dict(type=kind,base_ids=x,edited_ids=y,
                              base_text=a,edited_text=b))
            made+=1
        print('generated',kind,made,'tries',tries,flush=True)
    return panel


def run():
    torch.set_num_threads(4)
    gate=json.loads((ART/'train_finite_fidelity_gate.json').read_text())
    selected=gate['selected']
    assert selected['method']=='pooled_edit'
    layer,head,rank=gate['layer'],gate['head'],selected['rank']
    panel=generate(Tokenizer.from_file(TOKENIZER))
    scan=np.load(ART/'all_head_unlabeled_edit_scan.npz')
    train=scan[f'layer{layer}__train'][:,head].astype(np.float64)
    basis=torch.tensor(top_basis(train.T@train/len(train))[:,:rank].copy(),
                       dtype=torch.float32)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probe=HeadCut(model.transformer.h[layer].attn);probe.head=head
    groups=[]
    for length in sorted({len(r['base_ids']) for r in panel}):
        indices=[i for i,r in enumerate(panel) if len(r['base_ids'])==length]
        groups.extend(indices[j:j+8] for j in range(0,len(indices),8))
    rows=[]
    with torch.no_grad():
        for indices in groups:
            before=torch.tensor([panel[i]['base_ids'] for i in indices],dtype=torch.long)
            after=torch.tensor([panel[i]['edited_ids'] for i in indices],dtype=torch.long)
            probe.basis=None;probe.injection=None
            final_logits(model,before,stop_at=layer)
            base_state=probe.last_input[:,-1,head*128:(head+1)*128].clone()
            final_logits(model,after,stop_at=layer)
            delta=probe.last_input[:,-1,head*128:(head+1)*128]-base_state
            probe.basis=torch.eye(128)
            native_base=final_logits(model,before).double()
            native_edited=final_logits(model,after).double()
            probe.injection=delta
            native_patched=final_logits(model,before).double()
            probe.injection=None
            native_effect=native_patched-native_base
            target=native_edited-native_base
            probe.basis=basis
            reduced_base=final_logits(model,before).double()
            probe.injection=(delta@basis)@basis.T
            reduced_patched=final_logits(model,before).double()
            probe.injection=None
            reduced_effect=reduced_patched-reduced_base
            mse=(reduced_effect-native_effect).square().sum(-1)/native_effect.square().sum(-1)
            p=torch.log_softmax(native_base,dim=-1)
            q=torch.log_softmax(reduced_base,dim=-1)
            kl=(p.exp()*(p-q)).sum(-1)
            aligned=(native_effect*target).sum(-1)/target.square().sum(-1)
            cosine=(native_effect*target).sum(-1)/(
                native_effect.norm(dim=-1)*target.norm(dim=-1)).clamp_min(1e-30)
            for j,i in enumerate(indices):
                rows.append(dict(index=i,type=panel[i]['type'],
                    relative_path_mse=float(mse[j]),base_forward_kl=float(kl[j]),
                    native_path_l2=float(native_effect[j].norm()),
                    input_edit_l2=float(target[j].norm()),
                    native_path_target_projection=float(aligned[j]),
                    native_path_target_cosine=float(cosine[j])))
            print('batch',len(rows),'/',len(panel),flush=True)
    family={kind:dict(mean_path_mse=float(np.mean([r['relative_path_mse']
        for r in rows if r['type']==kind])),mean_native_target_cosine=float(
        np.mean([r['native_path_target_cosine'] for r in rows if r['type']==kind])))
        for kind in FAMILIES}
    result=dict(description='Frozen training-selected L9H7 rank96 pooled-edit cut on second unseen designed-family stress panel. 12 unique one-token edits per six families. Finite final-logit head-patch path MSE, ordinary forward KL, and native head-path alignment with whole input-edit logit response; no answer labels. Families excluded from fitting and selection.',
        layer=layer,head=head,rank=rank,method=selected['method'],seed=SEED,
        n_per_family=N_PER_FAMILY,cases=panel,rows=rows,family=family,
        max_family_path_mse=max(r['mean_path_mse'] for r in family.values()),
        mean_base_forward_kl=float(np.mean([r['base_forward_kl'] for r in rows])))
    (ART/'gate_second_family_stress.json').write_text(json.dumps(result,indent=2)+'\n')
    print('summary',family,'worst',result['max_family_path_mse'],
          'KL',result['mean_base_forward_kl'])


if __name__=='__main__':run()
