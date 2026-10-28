#!/usr/bin/env python3
"""Test source-specific residual corrections at matched total input-read budget."""
import json
from pathlib import Path
import numpy as np
import torch

from run_bilin18_l8_intervention_input_metric import collect,covariance,metric_form,evaluate,response,scalar,fidelity
from run_bilin18_l8_source_span_response import source_form,leading
from run_bilin18_qkov_causal_edges import load_full_model

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_hybrid_source_response'


def row_errors(targets,predictions,rows,positions):
    target=torch.stack(targets);pred=torch.stack(predictions)
    error=(target-pred).square().sum((0,2)).reshape(rows,positions).sum(1)
    energy=target.square().sum((0,2)).reshape(rows,positions).sum(1)
    return {'error':error.numpy(),'energy':energy.numpy()}


@torch.inference_mode()
def main():
    torch.set_num_threads(6)
    cfg=json.loads((ROOT/'configs/bilin18_l8_hybrid_source_response.json').read_text())
    frozen=np.load(ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_intervention_input_metric_frozen.npz')
    old=torch.from_numpy(frozen['old_inputs']);mean=torch.from_numpy(frozen['mean']);qs=list(torch.from_numpy(frozen['exact_forms']))
    axes=torch.from_numpy(frozen['axes']).cuda();cx=covariance(old-mean)
    model=load_full_model('cuda');model.embedding.float();model.blocks.float()
    weights=model.blocks[8].attn.c_proj.weight.detach().cpu().double().reshape(len(mean),model.config.n_head,-1)
    natural={rank:[metric_form(q,cx,rank) for q in qs] for rank in (64,128,256)}
    charts={}
    for rank in cfg['source_ranks']:
        charts[rank]={}
        for heads in [[h] for h in range(model.config.n_head)]+cfg['pair_removals']:
            key='_'.join(map(str,heads));charts[rank][key]={}
            for kind,chosen in [('correct',heads),('wrong',[(h+1)%model.config.n_head for h in heads])]:
                w=weights[:,chosen].reshape(len(mean),-1)
                charts[rank][key][kind]=leading(w@w.T,rank)
    # Only compact bases are saved; full quadratic matrices can be reconstructed.
    cache={f'rank{rank}_{key}_{kind}':b.numpy() for rank,items in charts.items() for key,rows in items.items() for kind,b in rows.items()}
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**cache)
    STEM.with_name(STEM.name+'_frozen.json').write_text(json.dumps(cfg,indent=2))
    print('hybrid bases frozen before test',flush=True)
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r')
    test,coef,replay=collect(model,corpus,cfg['test_start'],cfg,axes,test=True)
    groups={'fit_heads':[f'drop{h}' for h in cfg['fit_heads']], 'heldout_heads':[f'drop{h}' for h in cfg['heldout_heads']], 'half':[f'half{h}' for h in range(model.config.n_head)], 'pairs':[f'pair{a}_{b}' for a,b in cfg['pair_removals']]}
    results={f'natural{rank}':evaluate(test,coef,forms,qs,mean,old,groups) for rank,forms in natural.items()}
    results['exact']=evaluate(test,coef,qs,qs,mean,old,groups)
    row_stats={}
    for rank,forms in natural.items():
        row_stats[f'natural{rank}']={}
        for group,keys in groups.items():
            pred=[torch.stack([response(test['native'],test[key],q,a,mean) for q,a in zip(qs,forms)],-1) for key in keys]
            row_stats[f'natural{rank}'][group]=row_errors([coef[key]-coef['native'] for key in keys],pred,cfg['rows'],len(cfg['positions']))
    train_mean=torch.stack([((old@q)*old).sum(-1).mean() for q in qs]);x=test['native']
    for rank in cfg['source_ranks']:
        for kind in ('correct','wrong'):
            name=f'hybrid{4*rank}_{kind}';r={'groups':{},'natural_by_head':{}}
            predictions={}
            for key,rows in charts[rank].items():
                basis=rows[kind]
                forms=[a+source_form(q-a,basis) for q,a in zip(qs,natural[2*rank])]
                if '_' not in key:
                    pred=torch.stack([scalar(x,q,a,mean,old) for q,a in zip(qs,forms)],-1)
                    r['natural_by_head'][key]=fidelity(coef['native']-train_mean,pred-train_mean)
                targets=['pair'+key] if '_' in key else ['drop'+key,'half'+key]
                for target in targets:
                    predictions[target]=torch.stack([response(x,test[target],q,a,mean) for q,a in zip(qs,forms)],-1)
            r['natural_r2']=float(np.mean(list(r['natural_by_head'].values())))
            row_stats[name]={}
            for group,keys in groups.items():
                targets=[coef[key]-coef['native'] for key in keys];pred=[predictions[key] for key in keys]
                r['groups'][group]=fidelity(torch.cat(targets),torch.cat(pred))
                row_stats[name][group]=row_errors(targets,pred,cfg['rows'],len(cfg['positions']))
            results[name]=r;print(name,r,flush=True)
    gates={}
    for budget,threshold,gain in [(128,.8,.1),(256,.9,.05)]:
        c=results[f'hybrid{budget}_correct'];n=results[f'natural{budget}']
        gates[f'hybrid{budget}_response']=c['groups']['heldout_heads']>=threshold and c['groups']['heldout_heads']-n['groups']['heldout_heads']>=gain
        gates[f'hybrid{budget}_natural']=c['natural_r2']>=n['natural_r2']-.05
    gates['source_identity']=all(results[f'hybrid{b}_correct']['groups']['heldout_heads']-results[f'hybrid{b}_wrong']['groups']['heldout_heads']>=.1 for b in (128,256))
    gates['pair_generalization']=all(results[f'hybrid{b}_correct']['groups']['pairs']-results[f'natural{b}']['groups']['pairs']>=.05 for b in (128,256))
    gates['exact']=replay<=1e-5 and min(results['exact']['groups'].values())>=1-1e-8
    # Post-result uncertainty audit: resample complete text rows, retaining all
    # positions, heads, and output coordinates together. No model selection here.
    rng=np.random.default_rng(20261025);indices=rng.integers(cfg['rows'],size=(2000,cfg['rows']))
    boot={name:{group:1-value['error'][indices].sum(1)/value['energy'][indices].sum(1) for group,value in groups_.items()} for name,groups_ in row_stats.items()}
    intervals={name:{group:np.quantile(values,[.025,.975]).tolist() for group,values in groups_.items()} for name,groups_ in boot.items()}
    advantages={f'hybrid{b}_minus_{control}':{group:np.quantile(boot[f'hybrid{b}_correct'][group]-boot[control][group],[.025,.975]).tolist() for group in groups} for b in (128,256) for control in (f'natural{b}',f'hybrid{b}_wrong')}
    STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'gates':gates,'head_replay_rms':replay,'results':results,'posthoc_row_bootstrap_95':intervals,'posthoc_advantage_95':advantages},indent=2,allow_nan=False))
    np.savez_compressed(STEM.with_name(STEM.name+'_row_errors.npz'),**{f'{name}_{group}_{key}':value for name,groups_ in row_stats.items() for group,values in groups_.items() for key,value in values.items()})
    lines=['# L8 natural-state decoder plus source-specific response correction','',cfg['scope'],'',f'Locked gates: `{gates}`.','','| Method | Natural R2 | Fit heads | Held-out heads | Half | Pairs |','|---|---:|---:|---:|---:|---:|']
    for name,r in results.items():lines.append(f"| {name} | {r['natural_r2']:.3f} | {r['groups']['fit_heads']:.3f} | {r['groups']['heldout_heads']:.3f} | {r['groups']['half']:.3f} | {r['groups']['pairs']:.3f} |")
    lines+=['','Natural R2 for hybrid methods is averaged over nine source charts. The source identity is supplied by the intervention declaration, not inferred from model state. This experiment asks whether known source relations shorten a causal description; it does not establish a single executable forward feature circuit. Wrong-source assignment is one fixed control, not a permutation significance test.']
    lines+=['','## Post-result row bootstrap','','These 95% intervals resample 128 complete text rows (2,000 replicates), retaining all sites, heads and coordinates within each row. They quantify this panel’s row-sampling uncertainty, not model-selection uncertainty or corpus-wide guarantees.','','| Budget | Held-out response 95% interval | Advantage vs equal-budget global | Advantage vs wrong source |','|---|---|---|---|']
    for budget in (128,256):
        fmt=lambda x:f'[{x[0]:.3f}, {x[1]:.3f}]'
        lines.append(f"| {budget} | {fmt(intervals[f'hybrid{budget}_correct']['heldout_heads'])} | {fmt(advantages[f'hybrid{budget}_minus_natural{budget}']['heldout_heads'])} | {fmt(advantages[f'hybrid{budget}_minus_hybrid{budget}_wrong']['heldout_heads'])} |")
    STEM.with_name('BILIN18_L8_HYBRID_SOURCE_RESPONSE.md').write_text('\n'.join(lines)+'\n')
    print(gates,flush=True)


if __name__=='__main__':main()
