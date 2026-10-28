#!/usr/bin/env python3
"""Fixed-rank, globally integrable decoders trained on values and finite responses."""
import json
import gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_intervention_input_metric import collect, covariance, metric_form
from run_bilin18_l8_source_support_factorial import collect_sources, norm_match

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_direct_objective_grid'

def qvalues(z,u,w):
    reads=torch.einsum('nd,adr->nar',z,u)
    return (reads.square()*w).sum(-1)

def qresponse(delta,mid,u,w):
    return 2*(torch.einsum('nd,adr->nar',delta,u)*torch.einsum('nd,adr->nar',mid,u)*w).sum(-1)

def fullvalues(z,q):
    return torch.einsum('nd,adj,nj->na',z,q,z)

def factors(forms):
    us=[];ws=[]
    for a in forms:
        v,e=torch.linalg.eigh(a)
        order=v.abs().argsort(descending=True)
        us.append(e[:,order]);ws.append(v[order])
    return torch.stack(us),torch.stack(ws)

def response_data(panel,keys,q,mean):
    x=panel['native'];d=torch.cat([panel[k]-x for k in keys]);m=torch.cat([(panel[k]+x)/2-mean for k in keys])
    lin=2*d@(q@mean).T
    target=lin+2*torch.einsum('nd,adj,nj->na',d,q,m)
    return d,m,lin,target

def move_panel(p):
    return {k:v.cuda().float() for k,v in p.items()}

@torch.no_grad()
def evaluate(panel,groups,u,w,q,mean,old,scales,details=False):
    z=panel['native']-mean
    exact=fullvalues(panel['native'],q)
    old_y=fullvalues(old,q).mean(0)
    pred=old_y+2*z@(q@mean).T+qvalues(z,u,w)-qvalues(old-mean,u,w).mean(0)
    err=(pred-exact).square();energy=(exact-old_y).square()
    out={'natural_r2':float(1-err.sum()/energy.sum()),'natural_axis_r2':(1-err.sum(0)/energy.sum(0)).tolist(),
         'natural_loss':float((err.mean(0)/scales[0]).mean()),'groups':{},'group_loss':{}}
    for name,keys in groups.items():
        d,m,lin,target=response_data(panel,keys,q,mean)
        err=(lin+qresponse(d,m,u,w)-target).square();energy=target.square()
        out['groups'][name]=float(1-err.sum()/energy.sum())
        out['group_loss'][name]=float((err.mean(0)/scales[1]).mean())
    return out

def save_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))

def main():
    torch.set_num_threads(6)
    cfg=json.loads((ROOT/'configs/bilin18_l8_direct_objective_grid.json').read_text())
    frozen=np.load(STEM.parent/'bilin18_l8_intervention_input_metric_frozen.npz')
    old64=torch.from_numpy(frozen['old_inputs']);mean64=torch.from_numpy(frozen['mean']);q64=torch.from_numpy(frozen['exact_forms'])
    old=old64.cuda().float();mean=mean64.cuda().float();q=q64.cuda().float();axes=torch.from_numpy(frozen['axes']).cuda().double()
    torch.manual_seed(cfg['seed'])
    model=load_full_model('cuda');model.embedding.float();model.blocks.float()
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r')
    train,_,replay_train=collect(model,corpus,cfg['discovery_start'],cfg,axes)
    validation,_,replay_val=collect(model,corpus,cfg['validation_start'],cfg,axes)
    # Free the model during optimization; test data are not collected until all fits freeze.
    del model;gc.collect();torch.cuda.empty_cache()
    train=move_panel(train);validation=move_panel(validation)
    keys=[f'drop{h}' for h in cfg['fit_heads']];groups={'even':keys}
    d,m,lin,target=response_data(train,keys,q,mean)
    z=old-mean;quad=fullvalues(z,q)
    scales=(fullvalues(old,q).var(0,unbiased=False).clamp_min(1e-15),target.square().mean(0).clamp_min(1e-15))
    target_quad=target-lin
    cx=covariance(old64-mean64)
    results={};saved={};baselines={}
    for rank in cfg['ranks']:
        forms=[metric_form(a,cx,rank) for a in q64]
        init_u,init_w=factors(forms)
        init_u=init_u[:,:,:rank]*init_w[:,:rank].abs().sqrt()[:,None,:]
        init_w=init_w[:,:rank].sign()
        init_u=init_u.cuda().float();init_w=init_w.cuda().float()
        baselines[str(rank)]=evaluate(validation,groups,init_u,init_w,q,mean,old,scales)
        saved[f'baseline_{rank}_u']=init_u.cpu().numpy();saved[f'baseline_{rank}_w']=init_w.cpu().numpy()
        for weight in cfg['response_weights']:
            for initialization in cfg['initializations']:
                name=f'r{rank}_lambda{weight}_{initialization}'
                # Shared sampling seed across objective conditions; a separate perturbation generator.
                torch.manual_seed(cfg['seed']+rank)
                u=init_u.clone()
                if initialization=='perturbed_spectral':
                    noise=torch.randn_like(u);u+=cfg['initialization_noise_relative']*noise*u.norm()/noise.norm()
                u=torch.nn.Parameter(u);w=torch.nn.Parameter(init_w.clone())
                optimizer=torch.optim.Adam([u,w],lr=cfg['learning_rate'])
                best=float('inf');best_state=None;history=[]
                for step in range(cfg['steps']+1):
                    if step%cfg['checkpoint_every']==0:
                        val=evaluate(validation,groups,u,w,q,mean,old,scales)
                        tr=evaluate(train,groups,u,w,q,mean,old,scales)
                        score=val['natural_loss']+weight*val['group_loss']['even']
                        history.append({'step':step,'selection_score':score,'validation':val,'train':tr})
                        if score<best:
                            best=score;best_state=(u.detach().clone(),w.detach().clone(),step)
                    if step==cfg['steps']:break
                    i=torch.randint(len(z),(cfg['minibatch'],),device='cuda');j=torch.randint(len(z),(cfg['minibatch'],),device='cuda')
                    err=qvalues(z[i],u,w)-qvalues(z[j],u,w)-(quad[i]-quad[j])
                    loss=(err.square().mean(0)/(2*scales[0])).mean()
                    if weight:
                        k=torch.randint(len(d),(cfg['minibatch'],),device='cuda')
                        loss=loss+weight*((qresponse(d[k],m[k],u,w)-target_quad[k]).square().mean(0)/scales[1]).mean()
                    optimizer.zero_grad();loss.backward();optimizer.step()
                bu,bw,step=best_state
                saved[name+'_u']=bu.cpu().numpy();saved[name+'_w']=bw.cpu().numpy()
                results[name]={'rank':rank,'response_weight':weight,'initialization':initialization,'selected_step':step,'history':history}
                print(name,'selected',step,'val',history[step//cfg['checkpoint_every']]['validation'],flush=True)
                save_json(STEM.with_name(STEM.name+'_training.json'),{'config':cfg,'results':results})
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**saved)
    save_json(STEM.with_name(STEM.name+'_frozen.json'),{'config':cfg,'results':results,'baseline_validation':baselines})
    print('ALL FITS FROZEN; collecting held-out test',flush=True)
    model=load_full_model('cuda');model.embedding.float();model.blocks.float()
    test,tcoef,replay_test=collect(model,corpus,cfg['test_start'],cfg,axes,test=True)
    pre,hw,replay_sources=collect_sources(model,corpus,{**cfg,'start':cfg['test_start']})
    dim=pre.shape[-1];rng=torch.Generator(device='cuda').manual_seed(cfg['seed']+100)
    for h in cfg['heldout_heads']:
        projection=model.blocks[8].attn.c_proj.weight.reshape(dim,9,-1)[:,h]
        gaussian=torch.randn(len(pre),projection.shape[1],generator=rng,device='cuda')@projection.T
        isotropic=torch.randn(pre.shape,generator=rng,device='cuda')
        for kind,payload in [('gaussian',gaussian),('isotropic',isotropic)]:
            test[f'{kind}{h}']=F.rms_norm(pre-norm_match(payload,hw[:,h]),(dim,)).detach().cpu().double()
    del model,pre,hw;gc.collect();torch.cuda.empty_cache()
    test=move_panel(test)
    exact_native=fullvalues(test['native'],q).double().cpu()
    native_replay=float(((exact_native-tcoef['native']).square().sum()/tcoef['native'].square().sum()).sqrt())
    response_error=0.;response_energy=0.
    for h in range(9):
        delta=test[f'drop{h}']-test['native'];mid=(test[f'drop{h}']+test['native'])/2-mean
        pred=(2*delta@(q@mean).T+2*torch.einsum('nd,adj,nj->na',delta,q,mid)).double().cpu()
        gold=tcoef[f'drop{h}']-tcoef['native'];response_error+=float((pred-gold).square().sum());response_energy+=float(gold.square().sum())
    testgroups={'even':keys,'odd':[f'drop{h}' for h in cfg['heldout_heads']],
                'half':[f'half{h}' for h in range(9)],'pairs':[f'pair{a}_{b}' for a,b in cfg['pair_removals']],
                'gaussian':[f'gaussian{h}' for h in cfg['heldout_heads']], 'isotropic':[f'isotropic{h}' for h in cfg['heldout_heads']]}
    tested={}
    for name in [f'baseline_{r}' for r in cfg['ranks']]+list(results):
        u=torch.from_numpy(saved[name+'_u']).cuda();w=torch.from_numpy(saved[name+'_w']).cuda()
        tested[name]=evaluate(test,testgroups,u,w,q,mean,old,scales)
        print('TEST',name,tested[name],flush=True)
    # Paired row bootstrap for objective-vs-baseline differences, including all runs.
    rng_np=np.random.default_rng(cfg['seed']);indices=rng_np.integers(cfg['rows'],size=(2000,cfg['rows']))
    rowarrays={};intervals={}
    d,m,lin,target=response_data(test,testgroups['odd'],q,mean)
    energy=target.square().sum(-1).reshape(4,cfg['rows'],4).sum((0,2)).cpu().numpy()
    rowarrays['odd_energy']=energy
    for name in tested:
        u=torch.from_numpy(saved[name+'_u']).cuda();w=torch.from_numpy(saved[name+'_w']).cuda()
        errors=(lin+qresponse(d,m,u,w)-target).square().sum(-1).reshape(4,cfg['rows'],4).sum((0,2)).cpu().numpy()
        rowarrays[name+'_odd_error']=errors
    for name,r in results.items():
        difference=rowarrays[f'baseline_{r["rank"]}_odd_error']-rowarrays[name+'_odd_error']
        intervals[name]=np.quantile(difference[indices].sum(1)/energy[indices].sum(1),[.025,.975]).tolist()
    np.savez_compressed(STEM.with_name(STEM.name+'_row_errors.npz'),**rowarrays)
    out={'config':cfg,'replay':{'head_sum':[replay_train,replay_val,replay_test,replay_sources],'native':native_replay,'response':(response_error/response_energy)**.5},'test':tested,'odd_advantage_95':intervals,
         'training':results,'baseline_validation':baselines}
    save_json(STEM.with_suffix('.json'),out)
    lines=['# Direct-objective information grid','','All 18 fits frozen before collecting test. Scores concern four fixed L8 coordinates, not the whole MLP.','','| Run | Selected step | Natural R2 | Even | Odd | Half | Pairs | Gaussian | Isotropic |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in tested.items():
        scores=[r['natural_r2']]+[r['groups'][g] for g in testgroups]
        lines.append('| '+name+' | '+str(results.get(name,{}).get('selected_step',0))+' | '+' | '.join(f'{s:.3f}' for s in scores)+' |')
    lines+=['','Each rank is per output scalar. The decoder is a single symmetric quadratic per scalar and has no source identity input. Exact mean-state linear terms and discovery mean calibration are retained. Objective weights use per-axis discovery scales; table scores pool unnormalized squared errors. Two nearby starts do not rule out optimization failures. Row-bootstrap intervals in JSON condition on trained models and fixed payload draws.']
    STEM.with_name('BILIN18_L8_DIRECT_OBJECTIVE_GRID.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
