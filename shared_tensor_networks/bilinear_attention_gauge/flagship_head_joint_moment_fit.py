"""Rank-four natural-output/edit-response moment mixture at selected heads."""

import json

import numpy as np
import torch

from flagship_bracket_group_competent_transfer import swap
from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_rank_pilot import HeadCut,contexts,final_logits
from flagship_matrix_free import CORPUS,SNAPSHOT
from flagship_output_aware_head_fit import objective,optimize,top_eigenvectors
from flagship_transfer import ART


SEED=20261118
TRAIN_PER_LENGTH=1024
LAMBDA_VALUES=(0.,.001,.01,.03,.1,.3,1.)
EDIT_TYPE={'bracket_closure':'bracket','subject_verb_agreement':'noun_number'}


def head_state(model,probe,ids,layer,head):
    final_logits(model,torch.tensor(ids,dtype=torch.long)[None,:],stop_at=layer)
    start=head*probe.head_dim
    return probe.last_input[0,-1,start:start+probe.head_dim].clone()


def run():
    torch.set_num_threads(4)
    rng=np.random.default_rng(SEED)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    weights=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                       weights_only=True,mmap=True)
    model.load_state_dict(weights);del weights
    model.requires_grad_(False)
    probes={layer:HeadCut(model.transformer.h[layer].attn)
            for layer in sorted({layer for layer,_ in TARGETS.values()})}
    natural={target:[] for target in TARGETS}
    with torch.no_grad():
        for length in (32,64):
            x=contexts(CORPUS,rng,TRAIN_PER_LENGTH,range(0,16000),length)
            for batch in x.split(8):
                final_logits(model,batch,stop_at=max(probes))
                for target,(layer,head) in TARGETS.items():
                    p=probes[layer];start=head*p.head_dim
                    natural[target].append(
                        p.last_input[:,-1,start:start+p.head_dim].numpy())
            print('natural',length,'captured',len(x),flush=True)
        meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
        test=json.loads((ART/'agreement_reduction_group_factorial.json').read_text())
        held_swaps={tuple(x['token_swap']) for x in test['rows']}
        edits={}
        for target,(layer,head) in TARGETS.items():
            pool=[x for x in meta['panels']['train'] if x['type']==EDIT_TYPE[target]]
            if target=='subject_verb_agreement':
                pool=[x for x in pool if swap(x['base_ids'],x['edited_ids']) not in held_swaps]
                assert len(pool)==44
            else:assert len(pool)==80
            p=probes[layer]
            edits[target]=np.stack([(head_state(model,p,x['edited_ids'],layer,head)-
                                     head_state(model,p,x['base_ids'],layer,head)).numpy()
                                    for x in pool]).astype(np.float64)
            print(target,'edit vectors',len(edits[target]),flush=True)
    old=np.load(ART/'output_aware_head_fit_bases.npz')
    bases={};moments={};records=[]
    for target,(layer,head) in TARGETS.items():
        z=np.concatenate(natural[target]).astype(np.float64)
        d=edits[target]
        c_nat=z.T@z/len(z)
        c_edit=d.T@d/len(d)
        p=probes[layer]
        w=p.native.__self__.weight.detach().numpy()[:,
            head*p.head_dim:(head+1)*p.head_dim].astype(np.float64)
        gram=w.T@w
        energy_nat=float(np.trace(gram@c_nat))
        energy_edit=float(np.trace(gram@c_edit))
        assert energy_nat>0 and energy_edit>0
        moments[f'{target}__natural']=c_nat
        moments[f'{target}__edit']=c_edit
        moments[f'{target}__gram']=gram
        start_bases={'weight':old[f'{target}__weight'],
                     'output_aware':old[f'{target}__output_aware_2048'],
                     'edit_pca':top_eigenvectors(c_edit,4)}
        for lam in LAMBDA_VALUES:
            mixed=(1-lam)*c_nat/energy_nat+lam*c_edit/energy_edit
            mixed=(mixed+mixed.T)/2
            ct=torch.tensor(mixed,dtype=torch.float64)
            gt=torch.tensor(gram,dtype=torch.float64)
            candidates=[]
            for name,start in start_bases.items():
                basis,loss=optimize(ct,gt,start,steps=350)
                candidates.append((loss,name,basis))
            candidates.sort(key=lambda x:x[0])
            loss,initial,basis=candidates[0]
            assert np.linalg.norm(basis.T@basis-np.eye(4))<1e-10
            key=f'{target}__lambda_{lam:g}'
            bases[key]=basis
            bt=torch.tensor(basis,dtype=torch.float64)
            natural_error=float(objective(torch.tensor(c_nat),gt,bt)/energy_nat)
            edit_error=float(objective(torch.tensor(c_edit),gt,bt)/energy_edit)
            records.append(dict(target=target,lambda_edit=lam,
                fit_natural_count=len(z),fit_edit_count=len(d),
                starting_point_selected=initial,training_mixed_loss=loss,
                starting_losses={name:float(value) for value,name,_ in candidates},
                normalized_natural_output_mse=natural_error,
                normalized_edit_output_mse=edit_error))
            print(target,'lambda',lam,'natural',round(natural_error,5),
                  'edit',round(edit_error,5),'start',initial,flush=True)
    np.savez_compressed(ART/'head_joint_moment_fit_bases.npz',**bases)
    np.savez_compressed(ART/'head_joint_moment_fit_moments.npz',**moments)
    result=dict(description='Rank-four orthogonal head cuts optimized for normalized mixture of 2048 natural FineWeb output moments and answer-free token-edit output moments; bracket edits 80, agreement edits 44 excluding all validation token-swap identities',
        seed=SEED,rank=4,lambda_values=LAMBDA_VALUES,
        fit_natural_lengths=[32,64],fit_natural_count_per_length=TRAIN_PER_LENGTH,
        fit_fineweb_docs=[0,16000],edit_types=EDIT_TYPE,
        objective='(1-lambda)*E||W(I-P)z||²/E||Wz||² + lambda*E||W(I-P)d||²/E||Wd||²',
        no_task_answer_labels_in_fit=True,
        agreement_fit_excludes_all_validation_token_swaps=True,records=records)
    (ART/'head_joint_moment_fit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':run()
