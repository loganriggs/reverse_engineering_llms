#!/usr/bin/env python3
"""Audit full-output response coverage and its nonlinear downstream mediation."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_qkov_causal_edges import load_full_model,attention_details
from run_bilin18_l8_intervention_input_metric import collect
from run_bilin18_unseeded_route_end_to_end_causality import empty_metric,update_metric,finish

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_response_population'

def top_basis(gram,rank):
    values,vectors=torch.linalg.eigh((gram+gram.T)/2)
    return vectors[:,values.argsort(descending=True)[:rank]].T.float().cuda()

def coverage(x,basis):
    energy=x.square().sum();return float((x@basis.T).square().sum()/energy.clamp_min(1e-30))

@torch.inference_mode()
def suffix(model,state,x0,bus,layer):
    dim=state.shape[-1];x=state
    for block in model.blocks[layer+1:]:
        mixed=block.lambdas[0]*x+block.lambdas[1]*x0
        detail=attention_details(block.attn,F.rms_norm(mixed,(dim,)),bus,model.config,model.tt);bus=detail['value_bus']
        pre=mixed+detail['write'];z=F.rms_norm(pre,(dim,));x=pre+block.mlp.Down(block.mlp.Left(z)*block.mlp.Right(z))+block.mlp.Down_bias
    raw=model.lm_head(F.rms_norm(x,(dim,))[:,-1]).float();return 30*torch.tanh(raw/30)

@torch.inference_mode()
def finite_panel(model,ids,head,bases,cfg):
    dim=model.config.n_embd;x0=F.rms_norm(model.embedding(ids),(dim,));x=x0;bus=None
    for layer,block in enumerate(model.blocks):
        mixed=block.lambdas[0]*x+block.lambdas[1]*x0
        detail=attention_details(block.attn,F.rms_norm(mixed,(dim,)),bus,model.config,model.tt);bus=detail['value_bus']
        pre=mixed+detail['write'];z=F.rms_norm(pre,(dim,));write=block.mlp.Down(block.mlp.Left(z)*block.mlp.Right(z))
        if layer!=cfg['layer']:
            x=pre+write+block.mlp.Down_bias;continue
        attended=torch.einsum('bhk,bkhd->bhd',detail['pattern'][:,:,-1],detail['mixed_value'])
        hw=torch.einsum('bhd,ohd->bho',attended,block.attn.c_proj.weight.reshape(dim,9,-1))
        changed_pre=pre.clone();changed_pre[:,-1]-=hw[:,head]
        cz=F.rms_norm(changed_pre[:,-1],(dim,));cw=block.mlp.Down(block.mlp.Left(cz)*block.mlp.Right(cz));delta=cw-write[:,-1]
        intact=pre+write+block.mlp.Down_bias;no_response=changed_pre+write+block.mlp.Down_bias
        changed_write=write.clone();changed_write[:,-1]=cw
        gold=changed_pre+changed_write+block.mlp.Down_bias
        states={'intact':intact,'no_response':no_response,'gold':gold}
        full=no_response.clone();full[:,-1]+=cw-write[:,-1];states['replay_full']=full
        for name,basis in bases.items():
            state=no_response.clone();state[:,-1]+=(delta@basis.T)@basis;states[name]=state
        logits={};names=list(states)
        for start in range(0,len(names),cfg['finite_condition_chunk']):
            chunk=names[start:start+cfg['finite_condition_chunk']];count=len(chunk)
            cbus=bus.repeat(count,1,1,1) if bus.ndim==4 else bus.repeat(count,1,1)
            values=suffix(model,torch.cat([states[n] for n in chunk]),x0.repeat(count,1,1),cbus,layer).reshape(count,len(ids),-1)
            logits.update({n:values[i] for i,n in enumerate(chunk)})
        return logits,delta
    raise ValueError('layer missing')

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());torch.manual_seed(cfg['seed'])
    model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[cfg['layer']]
    def write(x):
        parts=[]
        for z in x.split(256):
            z=z.cuda().float();parts.append(block.mlp.Down(block.mlp.Left(z)*block.mlp.Right(z)))
        return torch.cat(parts)
    oldfile=np.load(STEM.parent/'bilin18_residual_input_forms.npz');old=torch.from_numpy(oldfile['discovery_writes_L8']).cuda().float();mean=old.mean(0);centered=old-mean
    cache=np.load(STEM.parent/'bilin18_l8_coverage_grid_train_validation.npz')
    train={k:write(torch.from_numpy(cache[f'train_{k}'])) for k in ['native']+[f'drop{h}' for h in range(9)]}
    responses={h:train[f'drop{h}']-train['native'] for h in range(9)}
    grams={name:sum(responses[h].double().T@responses[h].double() for h in heads).cpu() for name,heads in [('response_even',cfg['fit_heads']),('response_all',list(range(9)))]}
    grams['natural']=(centered.double().T@centered.double()).cpu();down=block.mlp.Down.weight.double();grams['weight']=(down@down.T).cpu()
    maxima={name:top_basis(gram,max(cfg['ranks'])) for name,gram in grams.items()}
    atlas=np.load(STEM.parent/'bilin18_known_ontology_residual_atlas.npz');residual=torch.from_numpy(atlas['residual_basis_L8']).cuda().float()
    frozen=np.load(STEM.parent/'bilin18_l8_intervention_input_metric_frozen.npz');selected=torch.from_numpy(frozen['axes']).cuda().float()
    bases={'selected4':selected}
    for name,basis in maxima.items():
        for rank in cfg['ranks']:bases[f'{name}{rank}']=basis[:rank]
    for rank in cfg['ranks']:
        if rank<=len(residual):bases[f'residual{rank}']=residual[:rank]
    bases['random64']=torch.linalg.qr(torch.randn(1152,64,device='cuda')).Q.T
    ortho=max(float((b@b.T-torch.eye(len(b),device='cuda')).abs().max()) for b in bases.values())
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),mean=mean.cpu().numpy(),**{k:v.cpu().numpy() for k,v in bases.items()})
    print('Output bases FROZEN before test; orthogonality',ortho,flush=True)
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r')
    inputs,_,replay=collect(model,corpus,cfg['test_start'],{**cfg,'fit_heads':list(range(9))},selected.double(),test=False)
    test={k:write(v) for k,v in inputs.items()};delta={h:test[f'drop{h}']-test['native'] for h in range(9)}
    np.savez_compressed(STEM.with_name(STEM.name+'_full_test_writes.npz'),**{k:v.cpu().numpy() for k,v in test.items()})
    population={};head_energy={str(h):float(delta[h].square().sum()) for h in range(9)};selected_energy={str(h):float((delta[h]@selected.T).square().sum()) for h in range(9)}
    row_arrays={}
    for name,basis in bases.items():
        head={str(h):{'raw':coverage(delta[h],basis),'centered':coverage(delta[h]-delta[h].mean(0),basis)} for h in range(9)}
        pooled=torch.cat(list(delta.values()));population[name]={'natural_coverage':coverage(test['native']-mean,basis),'response_coverage':coverage(pooled,basis),'heads':head}
        for h in range(9):row_arrays[f'{name}_head{h}_captured']=(delta[h]@basis.T).square().sum(-1).reshape(cfg['rows'],len(cfg['positions'])).sum(1).cpu().numpy()
    for h in range(9):row_arrays[f'head{h}_energy']=delta[h].square().sum(-1).reshape(cfg['rows'],len(cfg['positions'])).sum(1).cpu().numpy()
    head_stats={str(h):{'full_energy_share':head_energy[str(h)]/sum(head_energy.values()),'selected_energy_share':selected_energy[str(h)]/sum(selected_energy.values()),'mean_energy_fraction':float(len(delta[h])*delta[h].mean(0).square().sum()/delta[h].square().sum())} for h in range(9)}
    save=lambda obj:STEM.with_suffix('.json').write_text(json.dumps(obj,indent=2,allow_nan=False))
    output={'config':cfg,'orthogonality_error':ortho,'head_replay':replay,'population':population,'head_statistics':head_stats}
    save(output);np.savez_compressed(STEM.with_name(STEM.name+'_population_rows.npz'),**row_arrays)
    print('Population', {n:(r['natural_coverage'],r['response_coverage']) for n,r in population.items()},flush=True)
    # Full suffix: every approximation sees identical upstream head deletion.
    finite_bases={name:bases[name] for name in cfg['finite_methods']};metrics={};finite_rows={};gold_replay=0.;effect_sizes={}
    for head in cfg['finite_heads']:
        metrics[str(head)]={name:empty_metric() for name in ['no_response','replay_full']+list(finite_bases)}
        head_sizes={'mlp_logit_energy':0.,'total_head_logit_energy':0.,'mlp_ce_energy':0.,'total_head_ce_energy':0.}
        for offset in range(0,cfg['finite_rows'],cfg['finite_batch']):
            row=cfg['finite_start']+offset
            for position in cfg['finite_positions']:
                ids=torch.as_tensor(np.array(corpus[row:row+cfg['finite_batch'],:position+1],dtype=np.int64),device='cuda');targets=torch.as_tensor(np.array(corpus[row:row+cfg['finite_batch'],position+1],dtype=np.int64),device='cuda')
                logits,local_delta=finite_panel(model,ids,head,finite_bases,cfg)
                local_key=f'h{head}_local_energy';finite_rows.setdefault(local_key,np.zeros(cfg['finite_rows']))
                finite_rows[local_key][offset:offset+len(ids)]+=local_delta.square().sum(-1).cpu().numpy()
                for name,basis in finite_bases.items():
                    local_key=f'h{head}_{name}_local_captured';finite_rows.setdefault(local_key,np.zeros(cfg['finite_rows']))
                    finite_rows[local_key][offset:offset+len(ids)]+=(local_delta@basis.T).square().sum(-1).cpu().numpy()
                gold=logits['gold'];zero=logits['no_response'];intact=logits['intact'];gold_replay=max(gold_replay,float((gold-logits['replay_full']).abs().max()))
                effect=(gold-zero).square().sum(-1);head_sizes['mlp_logit_energy']+=float(effect.sum());head_sizes['total_head_logit_energy']+=float((gold-intact).square().sum())
                ce_gold=F.cross_entropy(gold,targets,reduction='none');ce_zero=F.cross_entropy(zero,targets,reduction='none');ce_intact=F.cross_entropy(intact,targets,reduction='none')
                ce_effect=(ce_gold-ce_zero).square();head_sizes['mlp_ce_energy']+=float(ce_effect.sum());head_sizes['total_head_ce_energy']+=float((ce_gold-ce_intact).square().sum())
                for name,metric in metrics[str(head)].items():
                    update_metric(metric,gold,zero,logits[name],targets)
                    key=f'h{head}_{name}';finite_rows.setdefault(key+'_error',np.zeros(cfg['finite_rows']));finite_rows[key+'_error'][offset:offset+len(ids)]+=(logits[name]-gold).square().sum(-1).cpu().numpy()
                    finite_rows.setdefault(key+'_ce_error',np.zeros(cfg['finite_rows']));finite_rows[key+'_ce_error'][offset:offset+len(ids)]+=(F.cross_entropy(logits[name],targets,reduction='none')-ce_gold).square().cpu().numpy()
                for suffix_name,vals in [('energy',effect),('ce_energy',ce_effect)]:
                    key=f'h{head}_{suffix_name}';finite_rows.setdefault(key,np.zeros(cfg['finite_rows']));finite_rows[key][offset:offset+len(ids)]+=vals.cpu().numpy()
            print('finite head',head,'rows',offset+cfg['finite_batch'],flush=True)
        effect_sizes[str(head)]=head_sizes
    finite={h:{n:finish(m) for n,m in methods.items()} for h,methods in metrics.items()}
    for h in finite:
        for name in finite_bases:
            finite[h][name]['same_site_local_response_coverage']=float(finite_rows[f'h{h}_{name}_local_captured'].sum()/finite_rows[f'h{h}_local_energy'].sum())
    rng=np.random.default_rng(cfg['seed']);ix=rng.integers(cfg['finite_rows'],size=(2000,cfg['finite_rows']));intervals={}
    for h,methods in finite.items():
        for name in methods:
            key=f'h{h}_{name}';intervals[key]={}
            for kind,errsuffix,ensuffix in [('logits','_error','_energy'),('ce','_ce_error','_ce_energy')]:
                err=finite_rows[key+errsuffix];energy=finite_rows[f'h{h}'+ensuffix]
                intervals[key][kind]=np.quantile(1-err[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()
    output.update({'finite':finite,'finite_bootstrap_95':intervals,'finite_effect_sizes':effect_sizes,'finite_full_replay_max_error':gold_replay});save(output)
    np.savez_compressed(STEM.with_name(STEM.name+'_finite_rows.npz'),**finite_rows)
    lines=['# Full L8 response population and nonlinear suffix coverage','','Output projections use exact native responses: these are communication/sufficiency measurements, not executable input decoders.','','| Output basis | Natural write coverage | All-head response coverage | Head1 | Head3 |','|---|---:|---:|---:|---:|']
    for n,r in population.items():lines.append(f"| {n} | {r['natural_coverage']:.3f} | {r['response_coverage']:.3f} | {r['heads']['1']['raw']:.3f} | {r['heads']['3']['raw']:.3f} |")
    lines+=['','## Nonlinear suffix response fidelity','','The denominator is native-response minus no-response with the SAME destination head removal. The intact model is a separate reference.','','| Basis | Head1 logits | Head1 CE | Head3 logits | Head3 CE |','|---|---:|---:|---:|---:|']
    for n in ['no_response','replay_full']+list(finite_bases):lines.append(f"| {n} | {finite['1'][n]['causal_fidelity']:.3f} | {finite['1'][n]['ce_effect_fidelity']:.3f} | {finite['3'][n]['causal_fidelity']:.3f} | {finite['3'][n]['ce_effect_fidelity']:.3f} |")
    lines+=['',f'Projection orthogonality error {ortho:.3g}; native head replay {replay:.3g}; full-response suffix replay max logit error {gold_replay:.3g}.','', 'Finite intervals in JSON resample32 complete rows, each containing2 positions. Ratios of mediation effect energy to total head effect energy are scale diagnostics, not additive causal shares; cancellation can make them exceed1. No semantic feature or unique basis is identified.']
    STEM.with_name(STEM.name.upper()+'.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
