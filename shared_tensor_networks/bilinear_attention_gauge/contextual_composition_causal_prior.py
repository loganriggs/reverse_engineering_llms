"""Finite-intervention training prior for the same sparse head dictionary.

Freeze the competent 16-task attention model. Fine-tune a 12-atom top-four
autoencoder on primitive+AND head states to preserve native logits and/or
finite single-key edit effects. XOR queries are absent from all fitting and
selection and are evaluated only on independent test sequences.
"""

import copy
import json

import numpy as np
import torch
from torch.nn import functional as F

from contextual_composition_reduction import FIT_QUERIES, exact_head, head_table
from contextual_composition_shared_groups import FlatSparse, hard_topk
from dual_joint_function_canonical import HERE
from train_contextual_composition_attention import (
    ContextualCompositionAttention, FAMILIES, examples)


OBJECTIVES = {
    'reconstruction_baseline': (0., 0.),
    'output_beta0.1': (.1, 0.),
    'output_beta0.3': (.3, 0.),
    'output_beta1': (1., 0.),
    'edit_gamma0.1': (0., .1),
    'edit_gamma0.3': (0., .3),
    'edit_gamma1': (0., 1.),
    'both_beta0.1_gamma0.1': (.1, .1),
    'both_beta0.3_gamma0.3': (.3, .3),
}
METHODS = (*OBJECTIVES, 'physical_top3', 'physical_top4')


class PhysicalTopK(torch.nn.Module):
    def __init__(self,k):
        super().__init__();self.k=k

    def forward(self,h):
        return h*hard_topk(h.abs(),self.k),{}


def toggle_one_key_bit(x, seed):
    rng=np.random.default_rng(seed)
    edited=x.clone()
    position=rng.integers(0,7,len(x))
    bit=rng.integers(0,4,len(x))
    edited[np.arange(len(x)),position] ^= torch.from_numpy(1<<bit)
    assert torch.all(edited[:,-1]==x[:,-1])
    return edited


def make_panel(native,table,n,seed,fit_only=False,shifted=False):
    probability=None
    if shifted:
        probability=np.arange(1,17,dtype=float)
        probability/=probability.sum()
    x,y,query=examples(n,seed,token_prob=probability)
    if fit_only:
        mask=np.isin(query,FIT_QUERIES)
        x,y,query=x[mask],y[mask],query[mask]
        assert not np.isin(query,FAMILIES['xor']).any()
    changed=toggle_one_key_bit(x,seed+100000)
    h=torch.from_numpy(exact_head(table,x)).float()
    h_edit=torch.from_numpy(exact_head(table,changed)).float()
    with torch.no_grad():
        q=native.embed(x[:,-1])
        original=native.read(torch.cat((h,q),dim=1)).squeeze(-1)
        edited=native.read(torch.cat((h_edit,q),dim=1)).squeeze(-1)
    return dict(x=x,changed_x=changed,h=h,h_edit=h_edit,q=q,
                original=original,edited=edited,y=y,query=query)


def outputs(featurizer,native,panel,indices=None):
    h=panel['h'] if indices is None else panel['h'][indices]
    he=panel['h_edit'] if indices is None else panel['h_edit'][indices]
    q=panel['q'] if indices is None else panel['q'][indices]
    rh,_=featurizer(h)
    rhe,_=featurizer(he)
    out=native.read(torch.cat((rh,q),dim=1)).squeeze(-1)
    out_edit=native.read(torch.cat((rhe,q),dim=1)).squeeze(-1)
    return rh,rhe,out,out_edit


def losses(featurizer,native,panel,indices=None):
    h=panel['h'] if indices is None else panel['h'][indices]
    he=panel['h_edit'] if indices is None else panel['h_edit'][indices]
    original=panel['original'] if indices is None else panel['original'][indices]
    edited=panel['edited'] if indices is None else panel['edited'][indices]
    rh,rhe,out,out_edit=outputs(featurizer,native,panel,indices)
    head=(F.mse_loss(rh,h)+F.mse_loss(rhe,he))/(
        h.square().mean()+he.square().mean()).clamp_min(1e-12)
    output=(F.mse_loss(out,original)+F.mse_loss(out_edit,edited))/(
        original.square().mean()+edited.square().mean()).clamp_min(1e-12)
    native_edit=edited-original
    model_edit=out_edit-out
    edit=F.mse_loss(model_edit,native_edit)/native_edit.square().mean().clamp_min(1e-12)
    return head,output,edit


def train_one(featurizer,native,train,val,beta,gamma,seed,steps=800):
    rng=np.random.default_rng(seed)
    opt=torch.optim.Adam(featurizer.parameters(),lr=.003)
    with torch.no_grad():
        initial=tuple(float(x) for x in losses(featurizer,native,val))
    baseline_head=initial[0]
    head_gate=baseline_head*1.05
    best=(initial[0]+beta*initial[1]+gamma*initial[2],
          copy.deepcopy(featurizer.state_dict()),0,initial)
    history=[]
    for step in range(1,steps+1):
        idx=torch.from_numpy(rng.integers(0,len(train['h']),512))
        head,output,edit=losses(featurizer,native,train,idx)
        loss=head+beta*output+gamma*edit
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(featurizer.parameters(),5.)
        opt.step()
        if step%50==0 or step==steps:
            with torch.no_grad():
                current=tuple(float(x) for x in losses(featurizer,native,val))
            score=current[0]+beta*current[1]+gamma*current[2]
            eligible=current[0]<=head_gate
            history.append(dict(step=step,head=current[0],output=current[1],
                                edit=current[2],eligible=eligible))
            if eligible and score<best[0]:
                best=(score,copy.deepcopy(featurizer.state_dict()),step,current)
    featurizer.load_state_dict(best[1])
    return dict(gate_passed=True,improved=best[2]>0,baseline_head=baseline_head,
                head_gate=head_gate,selected_step=best[2],
                selected_validation=dict(zip(('head','output','edit'),best[3])),
                history=history)


def evaluate(featurizer,native,panel):
    with torch.no_grad():
        rh,rhe,out,out_edit=outputs(featurizer,native,panel)
    h,he=panel['h'],panel['h_edit']
    original,edited=panel['original'],panel['edited']
    native_delta=edited-original
    delta=out_edit-out
    rows={}
    for family,ids in FAMILIES.items():
        mask=torch.from_numpy(np.isin(panel['query'],ids))
        rows[family]=dict(n=int(mask.sum()),
            accuracy=float(((out[mask]>0)==panel['y'][mask].bool()).float().mean()),
            head_relative_mse=float(((rh[mask]-h[mask]).square().sum()+
                                     (rhe[mask]-he[mask]).square().sum())/
                                    (h[mask].square().sum()+he[mask].square().sum())),
            relative_logit_mse=float(((out[mask]-original[mask]).square().sum()+
                                      (out_edit[mask]-edited[mask]).square().sum())/
                                     (original[mask].square().sum()+
                                      edited[mask].square().sum()).clamp_min(1e-12)),
            edit_path_relative_mse=float((delta[mask]-native_delta[mask]).square().sum()/
                                         native_delta[mask].square().sum().clamp_min(1e-12)),
            edit_sign_agreement=float(((delta[mask]>0)==
                                       (native_delta[mask]>0)).float().mean()))
    return rows


def check():
    torch.set_num_threads(1)
    native=ContextualCompositionAttention()
    native.load_state_dict(torch.load(HERE/'contextual_composition_attention_seed0.pt',
                                      map_location='cpu',weights_only=True))
    native.eval();native.requires_grad_(False)
    panel=make_panel(native,head_table(native),96,20264001,fit_only=True)
    x,changed=panel['x'],panel['changed_x']
    assert torch.all(x[:,-1]==changed[:,-1])
    differing=x[:,:-1]!=changed[:,:-1]
    assert torch.all(differing.sum(1)==1)
    token_xor=(x[:,:-1]^changed[:,:-1])[differing]
    assert set(token_xor.tolist()).issubset({1,2,4,8})
    with torch.no_grad():
        direct_h,_=native.head_outputs(x)
        assert (direct_h-panel['h']).abs().max()<3e-5
        assert (native(x)-panel['original']).abs().max()<1e-4
        assert (native(changed)-panel['edited']).abs().max()<1e-4
    flat=FlatSparse(12)
    flat.load_state_dict(torch.load(HERE/'contextual_composition_groups_seed0_flat_12.pt',
                                    map_location='cpu',weights_only=True))
    edit=losses(flat,native,panel)[2]
    edit.backward()
    for parameter in (flat.encoder,flat.decoder):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm()>0
    print('Exact head/logit, one-bit edit, and finite-path gradient checks passed.')


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(5):
        native=ContextualCompositionAttention()
        native.load_state_dict(torch.load(HERE/f'contextual_composition_attention_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        native.eval();native.requires_grad_(False)
        table=head_table(native)
        train=make_panel(native,table,30000,20263500+seed,fit_only=True)
        val=make_panel(native,table,8000,20263600+seed,fit_only=True)
        test=make_panel(native,table,20000,20263700+seed)
        shifted=make_panel(native,table,20000,20263800+seed,shifted=True)
        baseline_state=torch.load(HERE/f'contextual_composition_groups_seed{seed}_flat_12.pt',
                                  map_location='cpu',weights_only=True)
        methods={}
        for index,(name,(beta,gamma)) in enumerate(OBJECTIVES.items()):
            featurizer=FlatSparse(12)
            featurizer.load_state_dict(baseline_state)
            if name=='reconstruction_baseline':
                fit=dict(gate_passed=True,selected_step=0,
                         selected_validation=dict(zip(('head','output','edit'),
                         [float(x.detach()) for x in losses(featurizer,native,val)])))
            else:
                fit=train_one(featurizer,native,train,val,beta,gamma,
                              20263900+seed*100+index)
            methods[name]=dict(beta=beta,gamma=gamma,fit=fit,
                test=evaluate(featurizer,native,test),
                shifted=evaluate(featurizer,native,shifted))
            if name!='reconstruction_baseline':
                torch.save(featurizer.state_dict(),
                           HERE/f'contextual_composition_causal_seed{seed}_{name}.pt')
            print(seed,name,'step',fit['selected_step'],
                  'xor',round(methods[name]['test']['xor']['accuracy'],4),
                  'edit',round(methods[name]['test']['xor']['edit_path_relative_mse'],4),
                  flush=True)
        for k in (3,4):
            name=f'physical_top{k}'
            physical=PhysicalTopK(k)
            methods[name]=dict(beta=None,gamma=None,fit=dict(selected_step=None),
                test=evaluate(physical,native,test),
                shifted=evaluate(physical,native,shifted))
        rows.append(dict(seed=seed,methods=methods))
    summary={}
    for name in METHODS:
        summary[name]={}
        for split in ('test','shifted'):
            summary[name][split]={family:{metric:float(np.mean([
                row['methods'][name][split][family][metric] for row in rows]))
                for metric in ('accuracy','head_relative_mse','relative_logit_mse',
                               'edit_path_relative_mse','edit_sign_agreement')}
                for family in FAMILIES}
        summary[name]['improved_seeds']=sum(row['methods'][name]['fit'].get('improved',False)
                                             for row in rows)
    record=dict(description='Frozen 16-task bilinear attention models, five seeds. Fine-tune the same 12-atom top-four autoencoder from the same unsupervised checkpoint with head reconstruction, optional native output distillation, and/or finite single-key-bit edit-path distillation. Only primitive+AND queries in training and validation; all XOR queries withheld until uniform and shifted tests. No task labels enter fit. Select validation checkpoint by objective subject to <=5% degradation in allowed-query head error relative to initial baseline. Finite input edit effects evaluated on independent tokens/queries/seeds.',
                objectives=OBJECTIVES,rows=rows,summary=summary)
    (HERE/'contextual_composition_causal_prior.json').write_text(
        json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':run()
