"""Finite head-path fidelity for matched-rank later-layer head reductions."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_head_holdout import TARGETS, pairs
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT, GPTConfig
from flagship_later_head_rank_pilot import HeadCut, capture_head_states, final_logits, RANKS
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


def margin(model,ids,a,b):
    value=final_logits(model,torch.tensor(ids,dtype=torch.long)[None,:])[0]
    return float(value[a]-value[b])


def run():
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probes={layer:HeadCut(model.transformer.h[layer].attn)
            for layer in sorted(set(layer for layer,_ in TARGETS.values()))}
    saved=np.load(ART/'later_head_rank_pilot_bases.npz')
    old=json.loads((ART/'all_head_explicit_task_head_holdout.json').read_text())
    previous={(r['family'],r['base_prompt'],r['edited_prompt']):r
              for r in old['rows']}
    test=[]
    with torch.no_grad():
        for family,base,edited,answer_a,answer_b in pairs():
            before=tokenizer.encode(base,add_special_tokens=False).ids
            after=tokenizer.encode(edited,add_special_tokens=False).ids
            ans_a=tokenizer.encode(answer_a,add_special_tokens=False).ids
            ans_b=tokenizer.encode(answer_b,add_special_tokens=False).ids
            assert len(ans_a)==len(ans_b)==1
            layer,head=TARGETS[family]
            a=capture_head_states(model,probes,before)[layer][head]
            b=capture_head_states(model,probes,after)[layer][head]
            target=float(previous[family,base,edited]['selected_last_position_effect'])
            test.append(dict(family=family,before=before,answer_a=ans_a[0],
                             answer_b=ans_b[0],delta=b-a,target=target))
        # Identity projection plus full path edit must reproduce the earlier
        # native-model last-position head patch, independent of script hooks.
        identity_errors=[]
        for item in test:
            layer,head=TARGETS[item['family']]
            p=probes[layer];p.head=head;p.basis=torch.eye(p.head_dim)
            base=margin(model,item['before'],item['answer_a'],item['answer_b'])
            p.injection=item['delta'][None,:]
            patched=margin(model,item['before'],item['answer_a'],item['answer_b'])
            p.injection=None;p.basis=None
            identity_errors.append(abs((patched-base)-item['target']))
        assert max(identity_errors)<2e-4,max(identity_errors)
        records=[]
        for family,(layer,head) in TARGETS.items():
            p=probes[layer];p.head=head
            part=[x for x in test if x['family']==family]
            w=p.native.__self__.weight.detach().double()[:,head*p.head_dim:(head+1)*p.head_dim]
            for method in ('weight','corpus','task_delta','mixed','random'):
                vectors=saved[f'{family}__{method}']
                for rank in RANKS:
                    basis=torch.tensor(vectors[:,:rank].copy(),dtype=torch.float32)
                    p.basis=basis
                    effects=[]
                    targets=[]
                    for item in part:
                        base=margin(model,item['before'],item['answer_a'],item['answer_b'])
                        projected=((item['delta'][None,:]@basis)@basis.T)
                        p.injection=projected
                        patched=margin(model,item['before'],item['answer_a'],item['answer_b'])
                        p.injection=None
                        effects.append(patched-base)
                        targets.append(item['target'])
                    effects=np.array(effects);targets=np.array(targets)
                    deltas=torch.stack([x['delta'] for x in part]).double()
                    projected=(deltas@basis.double())@basis.double().T
                    raw_delta_fraction=float(projected.square().sum()/deltas.square().sum())
                    output_delta_relative_mse=float(((deltas-projected)@w.T).square().sum()/((deltas@w.T).square().sum()))
                    records.append(dict(family=family,layer=layer,head=head,
                        method=method,rank=rank,n=len(part),
                        native_path_mean_abs=float(np.mean(np.abs(targets))),
                        cut_path_mean_abs=float(np.mean(np.abs(effects))),
                        raw_delta_energy_fraction=raw_delta_fraction,
                        output_delta_relative_mse=output_delta_relative_mse,
                        path_mae=float(np.mean(np.abs(effects-targets))),
                        path_relative_mae=float(np.mean(np.abs(effects-targets))/np.mean(np.abs(targets))),
                        sign_matches=int(np.sum(np.sign(effects)==np.sign(targets)))))
                    print(family,method,rank,'path relative',
                          round(records[-1]['path_relative_mae'],3),flush=True)
            p.basis=None
    result=dict(description='Holdout finite edited-minus-base last-position head-output path after orthogonal rank cut; compare with native full-vector patch',
                identity_max_abs_error=max(identity_errors),n_holdout_pairs=len(test),
                records=records,
                caveat='Path sufficiency, not necessity or additive mediation. Other paths remain active in both base and cut model.')
    path=ART/'later_head_path_fidelity.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('Saved',path,flush=True)


if __name__=='__main__':run()
