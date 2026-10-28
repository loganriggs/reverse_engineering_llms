#!/usr/bin/env python3
"""Separate example coverage, source coverage, and optimization duration."""
import gc
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_intervention_input_metric import collect,covariance,metric_form
from run_bilin18_l8_source_support_factorial import collect_sources,norm_match
from run_bilin18_l8_direct_objective_grid import qvalues,qresponse,fullvalues,factors,response_data,move_panel,evaluate,save_json

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_coverage_grid'

def main():
    torch.set_num_threads(6)
    cfg=json.loads((ROOT/'configs/bilin18_l8_coverage_grid.json').read_text())
    frozen=np.load(STEM.parent/'bilin18_l8_intervention_input_metric_frozen.npz')
    old64=torch.from_numpy(frozen['old_inputs']);mean64=torch.from_numpy(frozen['mean']);q64=torch.from_numpy(frozen['exact_forms'])
    old=old64.cuda().float();mean=mean64.cuda().float();q=q64.cuda().float();axes=torch.from_numpy(frozen['axes']).cuda().double()
    torch.manual_seed(cfg['seed'])
    model=load_full_model('cuda');model.embedding.float();model.blocks.float()
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r')
    train,_,replay_train=collect(model,corpus,cfg['discovery_start'],cfg,axes)
    validation,_,replay_val=collect(model,corpus,cfg['validation_start'],{**cfg,'rows':cfg['evaluation_rows']},axes)
    np.savez_compressed(STEM.with_name(STEM.name+'_train_validation.npz'),
        **{f'train_{k}':v.float().numpy() for k,v in train.items()},
        **{f'validation_{k}':v.float().numpy() for k,v in validation.items()})
    del model;gc.collect();torch.cuda.empty_cache()
    train=move_panel(train);validation=move_panel(validation)
    even=[f'drop{h}' for h in cfg['source_sets']['even']];odd=[f'drop{h}' for h in [1,3,5,7]]
    small={k:v[:128*len(cfg['positions'])] for k,v in train.items()}
    _,_,_,scaletarget=response_data(small,even,q,mean)
    scales=(fullvalues(old,q).var(0,unbiased=False).clamp_min(1e-15),scaletarget.square().mean(0).clamp_min(1e-15))
    # Odd validation results are diagnostic only; checkpoint selection always uses even heads.
    valgroups={'even':even,'odd':odd}
    z=old-mean;quad=fullvalues(z,q)
    init_u,init_w=factors([metric_form(a,covariance(old64-mean64),cfg['rank']) for a in q64])
    init_u=(init_u[:,:,:cfg['rank']]*init_w[:,:cfg['rank']].abs().sqrt()[:,None,:]).cuda().float()
    init_w=init_w[:,:cfg['rank']].sign().cuda().float()
    saved={'baseline_u':init_u.cpu().numpy(),'baseline_w':init_w.cpu().numpy()};results={}
    conditions=[(rows,source,1) for rows in cfg['row_counts'] for source in cfg['source_sets']]+[(128,'even',0)]
    for rows,source,weight in conditions:
        name=f'rows{rows}_{source}' if weight else 'value_only'
        panel={k:v[:rows*len(cfg['positions'])] for k,v in train.items()}
        keys=[f'drop{h}' for h in cfg['source_sets'][source]]
        d,m,lin,target=response_data(panel,keys,q,mean);target_quad=target-lin
        torch.manual_seed(cfg['seed'])
        u=torch.nn.Parameter(init_u.clone());w=torch.nn.Parameter(init_w.clone())
        optimizer=torch.optim.Adam([u,w],lr=cfg['learning_rate']);best=float('inf');best_state=None;history=[]
        for step in range(cfg['steps']+1):
            if step%cfg['checkpoint_every']==0:
                val=evaluate(validation,valgroups,u,w,q,mean,old,scales)
                tr=evaluate(panel,{'fitted':keys},u,w,q,mean,old,scales)
                score=val['natural_loss']+weight*val['group_loss']['even']
                history.append({'step':step,'selection_score':score,'validation':val,'train':tr})
                if score<best:
                    best=score;best_state=(u.detach().clone(),w.detach().clone(),step)
                if step in cfg['fixed_step_comparisons']:
                    saved[f'{name}_step{step}_u']=u.detach().cpu().numpy().copy();saved[f'{name}_step{step}_w']=w.detach().cpu().numpy().copy()
                if step%400==0:print(name,'step',step,'train response',tr['groups'],'validation',val['groups'],flush=True)
            if step==cfg['steps']:break
            i=torch.randint(len(z),(cfg['minibatch'],),device='cuda');j=torch.randint(len(z),(cfg['minibatch'],),device='cuda')
            err=qvalues(z[i],u,w)-qvalues(z[j],u,w)-(quad[i]-quad[j])
            loss=(err.square().mean(0)/(2*scales[0])).mean()
            if weight:
                k=torch.randint(len(d),(cfg['minibatch'],),device='cuda')
                loss+=weight*((qresponse(d[k],m[k],u,w)-target_quad[k]).square().mean(0)/scales[1]).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        bu,bw,step=best_state;saved[name+'_u']=bu.cpu().numpy();saved[name+'_w']=bw.cpu().numpy()
        results[name]={'rows':rows,'sources':source,'response_weight':weight,'selected_step':step,
                       'unique_response_examples':len(d) if weight else 0,'response_minibatch_draws':cfg['steps']*cfg['minibatch'] if weight else 0,'history':history}
        save_json(STEM.with_name(STEM.name+'_training.json'),{'config':cfg,'results':results})
        print(name,'FROZEN selected step',step,flush=True)
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**saved)
    save_json(STEM.with_name(STEM.name+'_frozen.json'),{'config':cfg,'training':results})
    print('ALL CHECKPOINTS FROZEN; collecting test',flush=True)
    model=load_full_model('cuda');model.embedding.float();model.blocks.float()
    ecfg={**cfg,'rows':cfg['evaluation_rows']}
    test,tcoef,replay_test=collect(model,corpus,cfg['test_start'],ecfg,axes,test=True)
    pre,hw,replay_sources=collect_sources(model,corpus,{**ecfg,'start':cfg['test_start']})
    dim=pre.shape[-1];rng=torch.Generator(device='cuda').manual_seed(cfg['seed']+100)
    for h in [1,3,5,7]:
        projection=model.blocks[8].attn.c_proj.weight.reshape(dim,9,-1)[:,h]
        gaussian=torch.randn(len(pre),projection.shape[1],generator=rng,device='cuda')@projection.T
        isotropic=torch.randn(pre.shape,generator=rng,device='cuda')
        for kind,payload in [('gaussian',gaussian),('isotropic',isotropic)]:
            test[f'{kind}{h}']=F.rms_norm(pre-norm_match(payload,hw[:,h]),(dim,)).detach().cpu().double()
    np.savez_compressed(STEM.with_name(STEM.name+'_test.npz'),**{f'input_{k}':v.float().numpy() for k,v in test.items()},**{f'coefficient_{k}':v.numpy() for k,v in tcoef.items()})
    del model,pre,hw;gc.collect();torch.cuda.empty_cache();test=move_panel(test)
    response_error=0.;response_energy=0.
    for h in range(9):
        delta=test[f'drop{h}']-test['native'];mid=(test[f'drop{h}']+test['native'])/2-mean
        pred=(2*delta@(q@mean).T+2*torch.einsum('nd,adj,nj->na',delta,q,mid)).double().cpu()
        gold=tcoef[f'drop{h}']-tcoef['native'];response_error+=float((pred-gold).square().sum());response_energy+=float(gold.square().sum())
    testgroups={'even':even,'odd':odd,'half':[f'half{h}' for h in range(9)],'pairs':[f'pair{a}_{b}' for a,b in cfg['pair_removals']],
                'gaussian':[f'gaussian{h}' for h in [1,3,5,7]],'isotropic':[f'isotropic{h}' for h in [1,3,5,7]]}
    tested={};rowarrays={};methods=[k[:-2] for k in saved if k.endswith('_u')]
    for name in methods:
        u=torch.from_numpy(saved[name+'_u']).cuda();w=torch.from_numpy(saved[name+'_w']).cuda()
        tested[name]=evaluate(test,testgroups,u,w,q,mean,old,scales)
        for group,keys in testgroups.items():
            d,m,lin,target=response_data(test,keys,q,mean)
            rowarrays[group+'_energy']=target.square().sum(-1).reshape(len(keys),cfg['evaluation_rows'],len(cfg['positions'])).sum((0,2)).cpu().numpy()
            rowarrays[name+'_'+group+'_error']=(lin+qresponse(d,m,u,w)-target).square().sum(-1).reshape(len(keys),cfg['evaluation_rows'],len(cfg['positions'])).sum((0,2)).cpu().numpy()
        print('TEST',name,tested[name],flush=True)
    np.savez_compressed(STEM.with_name(STEM.name+'_row_errors.npz'),**rowarrays)
    rng_np=np.random.default_rng(cfg['seed']);ix=rng_np.integers(cfg['evaluation_rows'],size=(2000,cfg['evaluation_rows']))
    contrasts={}
    pairs=[('more_text_even','rows512_even','rows128_even'),('more_text_all','rows512_all','rows128_all'),
           ('more_sources_128','rows128_all','rows128_even'),('more_sources_512','rows512_all','rows512_even')]
    for name in results:pairs.append(('longer_'+name,name+'_step1600',name+'_step400'))
    for label,a,b in pairs:
        contrasts[label]={}
        for group in testgroups:
            diff=rowarrays[b+'_'+group+'_error']-rowarrays[a+'_'+group+'_error'];energy=rowarrays[group+'_energy']
            contrasts[label][group]={'gain':float(diff.sum()/energy.sum()),'paired_row_bootstrap_95':np.quantile(diff[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
    output={'config':cfg,'replay':{'head_sum':[replay_train,replay_val,replay_test,replay_sources],'response':(response_error/response_energy)**.5},'training':results,'test':tested,'contrasts':contrasts}
    save_json(STEM.with_suffix('.json'),output)
    lines=['# Text and source coverage information grid','','Rank64 per scalar; common validation selection; all checkpoints frozen before fresh test. Odd heads are unseen sources only in even-head fits.','','| Fit | Step | Natural R2 | Even | Odd | Half | Pairs | Gaussian | Isotropic |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['baseline']+list(results)+[k for k in methods if '_step' in k]:
        r=tested[name];step=results.get(name,{}).get('selected_step',name.split('_step')[-1] if '_step' in name else 0)
        scores=[r['natural_r2']]+[r['groups'][g] for g in testgroups]
        lines.append('| '+name+' | '+str(step)+' | '+' | '.join(f'{x:.3f}' for x in scores)+' |')
    lines+=['','All conditions use the same natural-value data. Counts refer to source-response text rows, with four sites per row. Five versus nine sources changes support and unique response count, while optimizer updates/draws remain fixed. Both payload families are held out. Paired row-bootstrap intervals condition on one training realization and fixed payload draws. Four selected L8 outputs do not establish whole-model coverage.']
    STEM.with_name('BILIN18_L8_COVERAGE_GRID.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
