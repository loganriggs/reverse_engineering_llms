#!/usr/bin/env python3
"""Separate supervision on actual source content from supervision on its support."""
import gc,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_intervention_input_metric import collect
from run_bilin18_l8_source_support_factorial import collect_sources,norm_match
from run_bilin18_l8_direct_objective_grid import qvalues,qresponse,fullvalues,response_data,move_panel,evaluate,save_json

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_head3_payload_training'

@torch.no_grad()
def payload_panel(pre,hw,projection,seed):
    rng=torch.Generator(device=pre.device).manual_seed(seed);ref=hw[:,3];dim=pre.shape[-1]
    permutation=torch.randperm(len(pre),generator=rng,device=pre.device)
    payloads={'actual':ref,'shuffled':norm_match(ref[permutation],ref),
              'gaussian':norm_match(torch.randn(len(pre),projection.shape[1],generator=rng,device=pre.device)@projection.T,ref),
              'isotropic':norm_match(torch.randn(pre.shape,generator=rng,device=pre.device),ref)}
    error=max(float((p.norm(dim=-1)-ref.norm(dim=-1)).abs().max()/ref.norm(dim=-1).max()) for p in payloads.values())
    return {k:F.rms_norm(pre-p,(dim,)) for k,p in payloads.items()},error

def quota_indices(sites,draws,has_supplement,device):
    chunks=[torch.randint(sites,(draws,),device=device)+h*sites for h in range(5)]
    chunks.append(torch.randint(sites if has_supplement else 5*sites,(draws,),device=device)+(5*sites if has_supplement else 0))
    return torch.cat(chunks)

def main():
    torch.set_num_threads(6)
    cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
    frozen=np.load(STEM.parent/'bilin18_l8_intervention_input_metric_frozen.npz')
    old=torch.from_numpy(frozen['old_inputs']).cuda().float();mean=torch.from_numpy(frozen['mean']).cuda().float();q=torch.from_numpy(frozen['exact_forms']).cuda().float();axes=torch.from_numpy(frozen['axes']).cuda().double()
    cache=np.load(STEM.parent/'bilin18_l8_coverage_grid_train_validation.npz')
    train={k.removeprefix('train_'):torch.from_numpy(cache[k]).cuda() for k in cache.files if k.startswith('train_')}
    validation={k.removeprefix('validation_'):torch.from_numpy(cache[k]).cuda() for k in cache.files if k.startswith('validation_')}
    even=[f'drop{h}' for h in cfg['fit_heads']];valgroups={'even':even,'head3':['drop3'],'other_odd':['drop1','drop5','drop7']}
    small={k:v[:512] for k,v in train.items()};_,_,_,target=response_data(small,even,q,mean)
    scales=(fullvalues(old,q).var(0,unbiased=False).clamp_min(1e-15),target.square().mean(0).clamp_min(1e-15))
    prior=np.load(STEM.parent/'bilin18_l8_coverage_grid_frozen.npz');init_u=torch.from_numpy(prior['baseline_u']).cuda();init_w=torch.from_numpy(prior['baseline_w']).cuda()
    model=load_full_model('cuda');model.embedding.float();model.blocks.float();corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r')
    pre,hw,replay_train=collect_sources(model,corpus,cfg)
    assert torch.allclose(F.rms_norm(pre,(pre.shape[-1],)),train['native'],atol=2e-6,rtol=2e-6)
    projection=model.blocks[8].attn.c_proj.weight.reshape(pre.shape[-1],9,-1)[:,3]
    extras={};norm_errors=[]
    for rep in cfg['replicates']:
        extras[rep],error=payload_panel(pre,hw,projection,cfg['seed']+rep);norm_errors.append(error)
    assert torch.allclose(extras[0]['actual'],train['drop3'],atol=2e-6,rtol=2e-6)
    np.savez_compressed(STEM.with_name(STEM.name+'_raw_train.npz'),pre=pre.cpu().numpy(),heads=hw.cpu().numpy(),projection=projection.detach().cpu().numpy(),**{f'{rep}_{k}':v.cpu().numpy() for rep,p in extras.items() for k,v in p.items()})
    del model,pre,hw,projection;gc.collect();torch.cuda.empty_cache()
    z=old-mean;quad=fullvalues(z,q);results={};saved={'baseline_u':init_u.cpu().numpy(),'baseline_w':init_w.cpu().numpy()}
    if cfg.get('reuse_primary_controls'):
        primary=np.load(STEM.parent/'bilin18_l8_head3_payload_training_frozen.npz')
        primary_results=json.loads((STEM.parent/'bilin18_l8_head3_payload_training.json').read_text())['training']
    for rep in cfg['replicates']:
        actual_panel={**train,'supplement':extras[rep]['actual']}
        _,_,actual_linear,actual_target=response_data(actual_panel,['supplement'],q,mean)
        actual_quadratic_energy=(actual_target-actual_linear).square().mean(0)
        for condition in cfg['conditions']:
            name=f'{condition}_rep{rep}';panel=dict(train);keys=list(even)
            if cfg.get('reuse_primary_controls') and condition in ['even_only','actual']:
                for suffix in ['_u','_w','_step1600_u','_step1600_w']:saved[name+suffix]=primary[name+suffix]
                results[name]={**primary_results[name],'reused_primary_fit':True}
                continue
            if condition!='even_only':panel['supplement']=extras[rep][condition];keys.append('supplement')
            d,m,lin,target=response_data(panel,keys,q,mean);target_quad=target-lin
            supplement_weight=torch.ones(4,device='cuda')
            if cfg.get('energy_match_supplement') and condition!='even_only':
                supplement_weight=actual_quadratic_energy/target_quad[-len(train['native']):].square().mean(0).clamp_min(1e-15)
            torch.manual_seed(cfg['seed']+1000+rep)
            u=torch.nn.Parameter(init_u.clone());w=torch.nn.Parameter(init_w.clone());optimizer=torch.optim.Adam([u,w],lr=cfg['learning_rate'])
            best=float('inf');history=[];best_state=None
            for step in range(cfg['steps']+1):
                if step%cfg['checkpoint_every']==0:
                    val=evaluate(validation,valgroups,u,w,q,mean,old,scales)
                    tr=evaluate(panel,{'fitted':keys},u,w,q,mean,old,scales)
                    score=val['natural_loss']+val['group_loss']['even']
                    history.append({'step':step,'selection_score':score,'validation':val,'train':tr})
                    if score<best:best=score;best_state=(u.detach().clone(),w.detach().clone(),step)
                    if step%400==0:print(name,step,'val',val['groups'],flush=True)
                if step==cfg['steps']:break
                i=torch.randint(len(z),(cfg['natural_minibatch'],),device='cuda');j=torch.randint(len(z),(cfg['natural_minibatch'],),device='cuda')
                err=qvalues(z[i],u,w)-qvalues(z[j],u,w)-(quad[i]-quad[j]);loss=(err.square().mean(0)/(2*scales[0])).mean()
                k=quota_indices(len(train['native']),cfg['draws_per_source'],condition!='even_only','cuda')
                response_loss=(qresponse(d[k],m[k],u,w)-target_quad[k]).square()/scales[1]
                draw_weights=torch.ones_like(response_loss)
                draw_weights[-cfg['draws_per_source']:]=supplement_weight
                loss+=(response_loss*draw_weights).mean()
                optimizer.zero_grad();loss.backward();optimizer.step()
            saved[name+'_step1600_u']=u.detach().cpu().numpy().copy();saved[name+'_step1600_w']=w.detach().cpu().numpy().copy()
            bu,bw,step=best_state;saved[name+'_u']=bu.cpu().numpy();saved[name+'_w']=bw.cpu().numpy()
            results[name]={'condition':condition,'replicate':rep,'selected_step':step,'history':history,'supplement_loss_weight':supplement_weight.tolist(),'training_response_energy_per_axis':target.square().mean(0).tolist()}
            save_json(STEM.with_name(STEM.name+'_training.json'),{'config':cfg,'training':results})
            print(name,'FROZEN',step,flush=True)
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**saved)
    save_json(STEM.with_name(STEM.name+'_frozen.json'),{'config':cfg,'training':results})
    print('ALL TEN FITS FROZEN; collecting test',flush=True)
    model=load_full_model('cuda');model.embedding.float();model.blocks.float();ecfg={**cfg,'rows':cfg['evaluation_rows']}
    test,tcoef,replay_test=collect(model,corpus,cfg['test_start'],ecfg,axes,test=True)
    pre,hw,replay_sources=collect_sources(model,corpus,{**ecfg,'start':cfg['test_start']})
    projection=model.blocks[8].attn.c_proj.weight.reshape(pre.shape[-1],9,-1)[:,3]
    for rep in cfg['replicates']:
        p,error=payload_panel(pre,hw,projection,cfg['seed']+100+rep);norm_errors.append(error)
        for kind,x in p.items():test[f'{kind}_draw{rep}']=x.cpu().double()
    test=move_panel(test)
    audit_error=0.;audit_energy=0.;block=model.blocks[8]
    with torch.no_grad():
        native=block.mlp.Down(block.mlp.Left(test['native'][:32])*block.mlp.Right(test['native'][:32])).double()@axes.T
        for key in [f'drop{h}' for h in range(9)]+[f'{k}_draw{r}' for r in cfg['replicates'] for k in ['shuffled','gaussian','isotropic']]:
            x=test[key][:32];gold=block.mlp.Down(block.mlp.Left(x)*block.mlp.Right(x)).double()@axes.T-native
            small={k:v[:32] for k,v in test.items()};_,_,_,target=response_data(small,[key],q,mean)
            audit_error+=float((target.double()-gold).square().sum());audit_energy+=float(gold.square().sum())
    np.savez_compressed(STEM.with_name(STEM.name+'_test.npz'),**{f'input_{k}':v.cpu().numpy() for k,v in test.items()},**{f'coefficient_{k}':v.numpy() for k,v in tcoef.items()})
    del model,pre,hw,projection,block;gc.collect();torch.cuda.empty_cache()
    groups={**valgroups,'shuffled':[f'shuffled_draw{r}' for r in cfg['replicates']],
            'gaussian':[f'gaussian_draw{r}' for r in cfg['replicates']], 'isotropic':[f'isotropic_draw{r}' for r in cfg['replicates']]}
    tested={};rows={}
    for name in [k[:-2] for k in saved if k.endswith('_u')]:
        u=torch.from_numpy(saved[name+'_u']).cuda();w=torch.from_numpy(saved[name+'_w']).cuda();tested[name]=evaluate(test,groups,u,w,q,mean,old,scales)
        for group,keys in groups.items():
            d,m,lin,target=response_data(test,keys,q,mean)
            rows[group+'_energy']=target.square().sum(-1).reshape(len(keys),cfg['evaluation_rows'],len(cfg['positions'])).sum((0,2)).cpu().numpy()
            rows[name+'_'+group+'_error']=(lin+qresponse(d,m,u,w)-target).square().sum(-1).reshape(len(keys),cfg['evaluation_rows'],len(cfg['positions'])).sum((0,2)).cpu().numpy()
        print('TEST',name,tested[name]['groups'],flush=True)
    np.savez_compressed(STEM.with_name(STEM.name+'_row_errors.npz'),**rows)
    ix=np.random.default_rng(cfg['seed']).integers(cfg['evaluation_rows'],size=(2000,cfg['evaluation_rows']));contrasts={}
    for name,r in results.items():
        contrasts[name]={}
        for ref in [f'even_only_rep{r["replicate"]}',f'actual_rep{r["replicate"]}']:
            contrasts[name][ref]={}
            for group in groups:
                diff=rows[ref+'_'+group+'_error']-rows[name+'_'+group+'_error'];energy=rows[group+'_energy']
                contrasts[name][ref][group]={'gain':float(diff.sum()/energy.sum()),'paired_row_bootstrap_95':np.quantile(diff[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
    def f(kind,rep,group='head3'):return tested[f'{kind}_rep{rep}']['groups'][group]
    gates={'generic_support':all(f(k,r)-f('even_only',r)>=.05 and f(k,r)>=f('actual',r)-.03 for r in cfg['replicates'] for k in ['shuffled','gaussian']),
           'context_pairing':all(f('actual',r)-f(k,r)>=.05 for r in cfg['replicates'] for k in ['shuffled','gaussian']),
           'generic_regularization':all(f('isotropic',r)>=f('actual',r)-.03 and f('isotropic',r,'isotropic')-f('even_only',r,'isotropic')>=.05 for r in cfg['replicates'])}
    save_json(STEM.with_suffix('.json'),{'config':cfg,'gates':gates,'replay':{'head_sum':[replay_train,replay_test,replay_sources],'finite_response':(audit_error/audit_energy)**.5,'norm_matching':norm_errors},'training':results,'test':tested,'contrasts':contrasts})
    lines=['# Head3 payload-supervision discriminator','','Ten fits frozen before test; selected on natural plus even-head validation only. All response fits receive the same number of gradient draws.','','| Fit | Selected step | Natural R2 | Even | Actual head3 | Other odd | Shuffled3 | Gaussian3 | Isotropic3 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['baseline']+list(results):
        r=tested[name];scores=[r['natural_r2']]+[r['groups'][g] for g in groups]
        lines.append('| '+name+' | '+str(results.get(name,{}).get('selected_step',0))+' | '+' | '.join(f'{s:.3f}' for s in scores)+' |')
    lines+=['',f'Locked gates: {gates}.','', 'All payloads are norm matched before RMS recomputation. Shuffled/Gaussian/isotropic test draws are independent from fixed training payloads. Two replicates vary SGD and training payload draws, not the text corpus. Full endpoint scores, histories and conditional row-bootstrap contrasts are in JSON. Four L8 coordinates remain a limited output population.']
    STEM.with_name(STEM.name.upper()+'.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    import sys
    if '--energy-control' in sys.argv:STEM=STEM.with_name('bilin18_l8_head3_energy_control')
    main()
