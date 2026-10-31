"""Rank-two reduction of trained attention under a nonlinear downstream readout.

The attention checkpoints are trained; the bilinear parity readout is analytic.
No task labels enter the reduction fits. Run on CPU.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from analyze_three_count_language import flatten,head_tables,logits
from dual_joint_function_canonical import HERE
from three_count_metric import pair_weights,weighted_fit
from three_count_task_fit import count_features
from train_three_count_language import ThreeCountAttention,examples


def svd_factors(matrix,rank=2):
    u,s,v=np.linalg.svd(matrix,full_matrices=False)
    return u[:,:rank]*np.sqrt(s[:rank]),np.sqrt(s[:rank,None])*v[:rank]


def context_arrays(model,seed,ntrain,nval,ntest):
    matrix=flatten(sum(head_tables(model)))
    bias=model.read.bias.detach().numpy()
    result={}
    for name,n,offset,holdout,rate in (
        ('train',ntrain,10000,True,.75),
        ('validation',nval,20000,True,.75),
        ('test',ntest,30000,False,.75),
        ('test_low_rate',10000,40000,False,.35),
        ('test_high_rate',10000,50000,False,1.2)):
        tokens,targets,attrs=examples(n,offset+seed,holdout,rate)
        tok=tokens.numpy()
        table_logits=logits(head_tables(model),tok,bias)
        with torch.no_grad():
            direct=model(tokens[:128]).detach().numpy()
        assert np.max(np.abs(direct-table_logits[:128]))<1e-10
        result[name]=dict(counts=count_features(tok).astype(np.float64),
            query=(tok[:,-1]-8).astype(np.int64),
            target=targets.numpy().astype(bool),attrs=attrs,
            teacher=table_logits)
    assert not result['validation']['attrs']['heldout'].any()
    assert result['test']['attrs']['heldout'].sum()>0
    return matrix,bias,result


def forward(matrix,bias,panel):
    table=matrix.reshape(2,3,8)
    return bias+np.einsum('nk,nok->no',panel['counts'],table[panel['query']])


def parity(logits):
    return -logits[:,0]*logits[:,1]


def loss_parts(pred,teacher,logit_scale,parity_scale):
    return (float(np.mean((pred-teacher)**2)/logit_scale),
            float(np.mean((parity(pred)-parity(teacher))**2)/parity_scale))


def fit_context(matrix,bias,train,val,nonlinear_weight,uniform_weight=0.,
                steps=1500,starts=2,seed=0):
    torch.set_num_threads(1)
    original_l,original_r=svd_factors(matrix)
    rng=np.random.default_rng(60000+seed)
    counts=torch.from_numpy(train['counts'])
    query=torch.from_numpy(train['query'])
    target=torch.from_numpy(train['teacher'])
    b=torch.from_numpy(bias)
    logit_scale=float(np.mean(train['teacher']**2))
    parity_scale=float(np.mean(parity(train['teacher'])**2))
    table_scale=float(np.mean(matrix**2))
    reference_table=torch.from_numpy(matrix)
    vcounts=torch.from_numpy(val['counts'])
    vquery=torch.from_numpy(val['query'])
    vtarget=torch.from_numpy(val['teacher'])
    def predict(l,r,c,q):
        table=(l@r).reshape(2,3,8)
        return b+torch.einsum('nk,nok->no',c,table[q])
    best=None;traces=[]
    for start in range(starts):
        noise=0. if start==0 else .15
        l=torch.nn.Parameter(torch.from_numpy(original_l+
            noise*rng.normal(size=original_l.shape)))
        r=torch.nn.Parameter(torch.from_numpy(original_r+
            noise*rng.normal(size=original_r.shape)))
        optimizer=torch.optim.Adam((l,r),lr=.01)
        best_start=None;trace=[]
        for step in range(1,steps+1):
            idx=torch.from_numpy(rng.integers(0,len(counts),1024))
            y=predict(l,r,counts[idx],query[idx]);t=target[idx]
            logit_loss=torch.mean((y-t)**2)/logit_scale
            nonlinear_loss=torch.mean(((-y[:,0]*y[:,1])-(-t[:,0]*t[:,1]))**2)/parity_scale
            table_loss=torch.mean(((l@r)-reference_table)**2)/table_scale
            objective=(logit_loss+nonlinear_weight*nonlinear_loss+
                       uniform_weight*table_loss)
            optimizer.zero_grad();objective.backward()
            torch.nn.utils.clip_grad_norm_((l,r),5.)
            optimizer.step()
            if step==1 or step%100==0 or step==steps:
                with torch.no_grad():
                    vy=predict(l,r,vcounts,vquery)
                    vl=float(torch.mean((vy-vtarget)**2)/logit_scale)
                    vp=float(torch.mean(((-vy[:,0]*vy[:,1])-
                        (-vtarget[:,0]*vtarget[:,1]))**2)/parity_scale)
                    vt=float(torch.mean(((l@r)-reference_table)**2)/table_scale)
                    vo=vl+nonlinear_weight*vp+uniform_weight*vt
                trace.append(dict(step=step,logit_mse=vl,
                                  nonlinear_mse=vp,table_mse=vt,objective=vo))
                if best_start is None or vo<best_start['objective']:
                    best_start=dict(matrix=(l@r).detach().numpy().copy(),
                                    step=step,objective=vo)
        traces.append(dict(start=start,trace=trace,
                           selected_step=best_start['step'],
                           selected_objective=best_start['objective']))
        if best is None or best_start['objective']<best['objective']:
            best=dict(start=start,**best_start)
    return best,traces


def evaluate(matrix,original,bias,panels,scales):
    test=panels['test'];pred=forward(matrix,bias,test)
    teacher=test['teacher'];target=test['target']
    bits=pred>0
    # Sign of -logit_0 * logit_1 is the XOR of the first two output bits.
    parity_target=np.logical_xor(target[:,0],target[:,1])
    parity_pred=parity(pred)>0
    parity_teacher=parity(teacher)>0
    assert np.all(parity_teacher==parity_target)
    results={}
    for name,mask in (('all',np.ones(len(pred),dtype=bool)),
                      ('common',~test['attrs']['rare']),
                      ('rare_heldout',test['attrs']['heldout'])):
        ll,pp=loss_parts(pred[mask],teacher[mask],*scales)
        results[name]=dict(n=int(mask.sum()),
            three_bit_joint_accuracy=float(np.mean(np.all(bits[mask]==target[mask],axis=1))),
            parity_accuracy=float(np.mean(parity_pred[mask]==parity_target[mask])),
            logit_relative_mse=ll,nonlinear_relative_mse=pp)
    results['uniform_table_relative_mse']=float(np.sum((matrix-original)**2)/
                                                 np.sum(original**2))
    results['count_rate_shift']={}
    for name in ('test_low_rate','test_high_rate'):
        panel=panels[name];shifted=forward(matrix,bias,panel)
        truth=panel['target'];rare=panel['attrs']['heldout']
        xor=np.logical_xor(truth[:,0],truth[:,1])
        results['count_rate_shift'][name]=dict(
            n=len(shifted),n_rare_heldout=int(rare.sum()),
            parity_accuracy=float(np.mean((parity(shifted)>0)==xor)),
            rare_heldout_parity_accuracy=float(np.mean(
                (parity(shifted[rare])>0)==xor[rare])),
            three_bit_joint_accuracy=float(np.mean(np.all(
                (shifted>0)==truth,axis=1))))
    return results


def run(seeds=(0,1,2),ntrain=20000,nval=5000,ntest=30000,steps=1500,starts=2):
    torch.set_num_threads(1)
    rows=[]
    for seed in seeds:
        model=ThreeCountAttention().double()
        model.load_state_dict(torch.load(HERE/f'three_count_seed{seed}.pt',
                             map_location='cpu',weights_only=True))
        model.eval()
        original,bias,panels=context_arrays(model,seed,ntrain,nval,ntest)
        scales=(float(np.mean(panels['train']['teacher']**2)),
                float(np.mean(parity(panels['train']['teacher'])**2)))
        u,s,v=np.linalg.svd(original,full_matrices=False)
        candidates={'native_table':original,
                    'uniform_table':(u[:,:2]*s[:2])@v[:2],
                    'uniform_table_rank3':(u[:,:3]*s[:3])@v[:3]}
        train_tokens,_,_=examples(ntrain,10000+seed,True)
        empirical=pair_weights(train_tokens.numpy())
        candidates['empirical_table'],_=weighted_fit(original,empirical,2)
        candidates['half_uniform_empirical_table'],_=weighted_fit(
            original,.5*(empirical+np.ones_like(empirical)/empirical.size),2)
        fits={}
        for weight,uniform_weight in ((0.,0.),(.05,0.),(.1,0.),(.3,0.),
                                      (1.,0.),(4.,0.),(1.,.1),(1.,1.),(1.,4.),
                                      (1.,20.),(1.,100.)):
            name=(f'context_nonlinear_weight_{weight:g}'+
                  (f'_uniform_weight_{uniform_weight:g}' if uniform_weight else ''))
            best,trace=fit_context(original,bias,panels['train'],
                                    panels['validation'],weight,uniform_weight,
                                    steps,starts,seed)
            candidates[name]=best['matrix']
            fits[name]=dict(selected_start=best['start'],
                selected_step=best['step'],validation_objective=best['objective'],
                traces=trace)
        candidate_scores={name:evaluate(m,original,bias,panels,scales)
                          for name,m in candidates.items()}
        rows.append(dict(seed=seed,fit=fits,scores=candidate_scores,
                         matrices={name:m.tolist() for name,m in candidates.items()}))
        print(seed,[(name,round(score['all']['parity_accuracy'],3),
                    round(score['rare_heldout']['parity_accuracy'],3))
                    for name,score in candidate_scores.items()],flush=True)
    output=dict(seeds=list(seeds),ntrain=ntrain,nval=nval,ntest=ntest,
        trained_part='saved three-head bilinear-attention checkpoints',
        analytic_readout='parity logit = -output_logit_0 * output_logit_1; sign is XOR of those two bits',
        fit_labels_used=False,rank=2,
        rank_exceptions=['native_table','uniform_table_rank3'],
        validation_excludes_rare_combination=True,
        count_rates=dict(train=.75,test=.75,test_low_rate=.35,test_high_rate=1.2),
        scope='exact first-layer token table plus analytic nonlinear readout, not a trained two-layer transformer',
        rows=rows)
    path=HERE/('three_count_nonlinear_metric_s'+
               '-'.join(map(str,seeds))+'.json')
    path.write_text(json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--ntrain',type=int,default=20000)
    p.add_argument('--nval',type=int,default=5000)
    p.add_argument('--ntest',type=int,default=30000)
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--starts',type=int,default=2)
    a=p.parse_args()
    run(tuple(a.seeds),a.ntrain,a.nval,a.ntest,a.steps,a.starts)
