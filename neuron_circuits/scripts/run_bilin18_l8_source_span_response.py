#!/usr/bin/env python3
"""Keep every quadratic interaction touching a shared or head-defined source span."""
import json
from pathlib import Path
import numpy as np
import torch

from run_bilin18_l8_intervention_input_metric import collect,covariance,metric_form,evaluate,response,scalar,fidelity
from run_bilin18_qkov_causal_edges import load_full_model

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_source_span_response'


def source_form(q,basis):
    """Exactly retain P-Q and Q-P cross interactions; at most 2r input reads."""
    p=basis@basis.T
    return p@q+q@p-p@q@p


def leading(c,rank):
    _,vectors=torch.linalg.eigh((c+c.T)/2)
    return vectors[:,-rank:]


@torch.inference_mode()
def main():
    torch.set_num_threads(6)
    cfg=json.loads((ROOT/'configs/bilin18_l8_source_span_response.json').read_text())
    frozen=np.load(ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_intervention_input_metric_frozen.npz')
    old=torch.from_numpy(frozen['old_inputs']);mean=torch.from_numpy(frozen['mean']);qs=list(torch.from_numpy(frozen['exact_forms']))
    axes=torch.from_numpy(frozen['axes']).cuda();cx=covariance(old-mean)
    model=load_full_model('cuda');model.embedding.float();model.blocks.float()
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r')
    discovery,_,dreplay=collect(model,corpus,cfg['discovery_start'],cfg,axes)
    deltas=torch.cat([discovery[f'drop{h}']-discovery['native'] for h in cfg['fit_heads']]);cd=covariance(deltas)
    weights=model.blocks[8].attn.c_proj.weight.detach().cpu().double().reshape(len(mean),model.config.n_head,-1)
    groups={'fit_heads':[f'drop{h}' for h in cfg['fit_heads']], 'heldout_heads':[f'drop{h}' for h in cfg['heldout_heads']], 'half':[f'half{h}' for h in range(model.config.n_head)], 'pairs':[f'pair{a}_{b}' for a,b in cfg['pair_removals']]}
    shared={};conditional={};cache={}
    for rank in cfg['source_ranks']:
        shared[f'natural{2*rank}']=[metric_form(q,cx,2*rank) for q in qs]
        basis=leading(cd,rank)
        shared[f'shared_source{rank}']=[source_form(q,basis) for q in qs]
        charts={}
        for heads in [[h] for h in range(model.config.n_head)]+cfg['pair_removals']:
            key='_'.join(map(str,heads));w=weights[:,heads].reshape(len(mean),-1)
            basis=leading(w@w.T,rank)
            charts[key]=[source_form(q,basis) for q in qs]
            cache[f'source{rank}_basis_{key}']=basis.numpy()
        conditional[f'conditional_source{rank}']=charts
    cache.update({name:torch.stack(forms).numpy() for name,forms in shared.items()})
    # Conditional forms reconstruct exactly from the saved bases and source Qs.
    # Avoid serializing a gigabyte of redundant full matrices.
    STEM.with_name(STEM.name+'_frozen.json').write_text(json.dumps(cfg,indent=2))
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**cache)
    print('source charts frozen before test',flush=True)
    test,coef,treplay=collect(model,corpus,cfg['test_start'],cfg,axes,test=True)
    results={name:evaluate(test,coef,forms,qs,mean,old,groups) for name,forms in shared.items()}
    x=test['native'];train_mean=torch.stack([((old@q)*old).sum(-1).mean() for q in qs])
    for name,charts in conditional.items():
        results[name]={'groups':{},'natural_by_head':{}}
        for key,forms in charts.items():
            if '_' not in key:
                pred=torch.stack([scalar(x,q,a,mean,old) for q,a in zip(qs,forms)],-1)
                results[name]['natural_by_head'][key]=fidelity(coef['native']-train_mean,pred-train_mean)
        results[name]['natural_r2']=float(np.mean(list(results[name]['natural_by_head'].values())))
        for group,keys in groups.items():
            targets=[];predictions=[]
            for key in keys:
                chart=key.removeprefix('drop').removeprefix('half').removeprefix('pair')
                targets.append(coef[key]-coef['native'])
                predictions.append(torch.stack([response(x,test[key],q,a,mean) for q,a in zip(qs,charts[chart])],-1))
            results[name]['groups'][group]=fidelity(torch.cat(targets),torch.cat(predictions))
    for name,value in results.items():print(name,value,flush=True)
    n=results['natural64'];c=results['conditional_source32'];s=results['shared_source32']
    gates={'source_transfer':c['groups']['heldout_heads']>=.8 and c['groups']['heldout_heads']-n['groups']['heldout_heads']>=.1,
           'family_transfer':all(c['groups'][key]-n['groups'][key]>=.1 for key in ('half','pairs')),
           'conditional_advantage':c['groups']['heldout_heads']-s['groups']['heldout_heads']>=.1,
           'natural_retention':c['natural_r2']>=n['natural_r2']-.05,
           'larger_budget':results['conditional_source64']['groups']['heldout_heads']>=.8 and results['conditional_source64']['groups']['heldout_heads']-results['natural128']['groups']['heldout_heads']>=.1}
    STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'gates':gates,'head_replay_rms':[dreplay,treplay],'results':results},indent=2,allow_nan=False))
    lines=['# L8 source-span response charts','',cfg['scope'],'',f'Locked gates: `{gates}`.','','| Method | Natural R2 | Fit heads | Held-out heads | Half | Pairs |','|---|---:|---:|---:|---:|---:|']
    for name,r in results.items():lines.append(f"| {name} | {r['natural_r2']:.3f} | {r['groups']['fit_heads']:.3f} | {r['groups']['heldout_heads']:.3f} | {r['groups']['half']:.3f} | {r['groups']['pairs']:.3f} |")
    lines+=['','Conditional natural R2 is the mean over single-head charts. Each chart is an integrable quadratic function, but choosing a different chart for each changed source does not recover a single global state function. Rank32 source charts and natural64 have the same maximum 65-read budget per output; rank64 source charts and natural128 have 129 reads. No decoder consumes the native write.']
    STEM.with_name('BILIN18_L8_SOURCE_SPAN_RESPONSE.md').write_text('\n'.join(lines)+'\n')
    print(gates,flush=True)


if __name__=='__main__':main()
