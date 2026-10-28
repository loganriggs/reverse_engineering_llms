#!/usr/bin/env python3
"""Compare state-compression and source-response input metrics at equal read budget."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_qkov_causal_edges import load_full_model, attention_details
from run_bilin18_residual_input_forms import contract, quadratic, sorted_eigh

ROOT = Path(__file__).resolve().parents[1]
STEM = ROOT / 'artifacts/bilin18_tensor_decomposition/bilin18_l8_intervention_input_metric'


def covariance(x):
    return x.T @ x / len(x)


def roots(c):
    values, vectors = torch.linalg.eigh((c+c.T)/2)
    values = values.clamp_min(values.max()*1e-8)
    return (vectors*values.sqrt())@vectors.T, (vectors*values.rsqrt())@vectors.T


def metric_form(q, metric, rank):
    s, inv = roots(metric)
    values, vectors = sorted_eigh(s@q@s)
    reads = inv@vectors[:,:rank]
    return (reads*values[:rank])@reads.T


def role_form(q, change_metric, context_metric, rank):
    sd, invd = roots(change_metric); sm, invm = roots(context_metric)
    u, values, vh = torch.linalg.svd(sd@q@sm, full_matrices=False)
    return (invd@u[:,:rank]*values[:rank])@(vh[:rank]@invm)


def scalar(x, q, approximation, mean, discovery):
    z=x-mean; d=discovery-mean
    intercept=quadratic(discovery,q).mean()-quadratic(d,approximation).mean()
    return intercept+2*z@q@mean+quadratic(z,approximation)


def response(x, changed, q, approximation, mean, secant=False):
    delta=changed-x; midpoint=(changed+x)/2-mean
    a=approximation if secant else (approximation+approximation.T)/2
    return 2*delta@q@mean+2*((delta@a)*midpoint).sum(-1)


def fidelity(target,prediction):
    return float(1-(target-prediction).square().sum()/target.square().sum().clamp_min(1e-30))


@torch.inference_mode()
def collect(model, corpus, start, cfg, axes, test=False):
    dim=model.config.n_embd; heads=model.config.n_head;hd=dim//heads
    interventions={f'drop{h}':([h],1.) for h in (range(heads) if test else cfg['fit_heads'])}
    if test:
        interventions.update({f'half{h}':([h],.5) for h in range(heads)})
        interventions.update({f'pair{a}_{b}':([a,b],1.) for a,b in cfg['pair_removals']})
    rows={key:[] for key in ['native']+list(interventions)}
    coefficients={key:[] for key in rows};sum_error=0.;sum_energy=0.
    for offset in range(0,cfg['rows'],cfg['batch_size']):
        ids=torch.as_tensor(np.array(corpus[start+offset:start+offset+cfg['batch_size'],:128],dtype=np.int64),device='cuda')
        x0=F.rms_norm(model.embedding(ids),(dim,));x=x0;bus=None
        for layer,block in enumerate(model.blocks):
            mixed=block.lambdas[0]*x+block.lambdas[1]*x0
            detail=attention_details(block.attn,F.rms_norm(mixed,(dim,)),bus,model.config,model.tt);bus=detail['value_bus']
            pre=mixed+detail['write'];norm=F.rms_norm(pre,(dim,))
            write=block.mlp.Down(block.mlp.Left(norm)*block.mlp.Right(norm))
            if layer==cfg['layer']:
                positions=cfg['positions'];native=norm[:,positions].flatten(0,1)
                rows['native'].append(native.cpu())
                coefficients['native'].append((write[:,positions].flatten(0,1).double()@axes.T).cpu())
                attended=torch.einsum('bhpk,bkhd->bphd',detail['pattern'][:,:,positions],detail['mixed_value'])
                hw=torch.einsum('bphd,ohd->bpho',attended,block.attn.c_proj.weight.reshape(dim,heads,hd))
                sum_error+=float((hw.sum(2)-detail['write'][:,positions]).square().sum());sum_energy+=float(detail['write'][:,positions].square().sum())
                for key,(selected,scale) in interventions.items():
                    changed=F.rms_norm(pre[:,positions]-scale*hw[:,:,selected].sum(2),(dim,)).flatten(0,1)
                    local=block.mlp.Down(block.mlp.Left(changed)*block.mlp.Right(changed)).double()@axes.T
                    rows[key].append(changed.cpu());coefficients[key].append(local.cpu())
                break
            x=pre+write+block.mlp.Down_bias
        if (offset+cfg['batch_size'])%32==0:print(f'panel {start}: {offset+cfg["batch_size"]}',flush=True)
    return {k:torch.cat(v).double() for k,v in rows.items()}, {k:torch.cat(v).double() for k,v in coefficients.items()}, float(np.sqrt(sum_error/sum_energy))


def perturbation_metrics(panel, heads, mask, mean):
    changes=[];midpoints=[]
    x=panel['native'][mask]
    for h in heads:
        changed=panel[f'drop{h}'][mask]
        changes.append(changed-x);midpoints.append((changed+x)/2-mean)
    return covariance(torch.cat(changes)),covariance(torch.cat(midpoints))


def evaluate(panel, coefficients, forms, qs, mean, discovery, groups, secant=False):
    x=panel['native'];pred=torch.stack([scalar(x,q,a,mean,discovery) for q,a in zip(qs,forms)],-1)
    train_mean=torch.stack([quadratic(discovery,q).mean() for q in qs])
    result={'natural_r2':fidelity(coefficients['native']-train_mean,pred-train_mean),'groups':{}}
    for group,keys in groups.items():
        targets=[];predictions=[]
        for key in keys:
            targets.append(coefficients[key]-coefficients['native'])
            predictions.append(torch.stack([response(x,panel[key],q,a,mean,secant) for q,a in zip(qs,forms)],-1))
        result['groups'][group]=fidelity(torch.cat(targets),torch.cat(predictions))
    return result


@torch.inference_mode()
def main():
    torch.set_num_threads(6)
    cfg=json.loads((ROOT/'configs/bilin18_l8_intervention_input_metric.json').read_text())
    torch.manual_seed(cfg['seed'])
    source=np.load(ROOT/cfg['source']);corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r')
    old=torch.from_numpy(source['discovery_inputs_L8']).double();mean=old.mean(0);cx=covariance(old-mean)
    axes=torch.from_numpy(source['directions_L8']).cuda().double()
    model=load_full_model('cuda');model.embedding.float();model.blocks.float()
    block=model.blocks[8]
    left,right,down=[m.weight.detach().cpu().double() for m in (block.mlp.Left,block.mlp.Right,block.mlp.Down)]
    qs=[contract(left,right,down,u) for u in axes.cpu()]
    discovery, dcoef, dreplay=collect(model,corpus,cfg['discovery_start'],cfg,axes)
    validation=(torch.arange(len(discovery['native']))//len(cfg['positions']))%4==0
    cd,cm=perturbation_metrics(discovery,cfg['fit_heads'],~validation,mean)
    cd=cd*torch.trace(cx)/torch.trace(cd)
    vp={k:v[validation] for k,v in discovery.items()};vc={k:v[validation] for k,v in dcoef.items()}
    groups={'fit_heads':[f'drop{h}' for h in cfg['fit_heads']]};selection={}
    for weight in cfg['mixture_weights']:
        forms=[metric_form(q,cx+weight*cd,cfg['rank']) for q in qs]
        result=evaluate(vp,vc,forms,qs,mean,old,groups)
        selection[str(weight)]={**result,'selection_score':min(result['natural_r2'],result['groups']['fit_heads'])}
        print('selection',weight,selection[str(weight)],flush=True)
    chosen=max(cfg['mixture_weights'],key=lambda w:selection[str(w)]['selection_score'])
    cd,cm=perturbation_metrics(discovery,cfg['fit_heads'],torch.ones(len(validation),dtype=torch.bool),mean)
    cd=cd*torch.trace(cx)/torch.trace(cd)
    methods={'natural64':[metric_form(q,cx,cfg['rank']) for q in qs],
             'mixed64':[metric_form(q,cx+chosen*cd,cfg['rank']) for q in qs],
             'raw64':[metric_form(q,torch.eye(len(mean),dtype=torch.float64),cfg['rank']) for q in qs]}
    role=[role_form(q,cd,cm,cfg['rank']//2) for q in qs]
    methods['symmetric_role32']=[(a+a.T)/2 for a in role]
    for seed in range(4):
        reads=torch.linalg.qr(torch.randn(len(mean),cfg['rank'],dtype=torch.float64)).Q
        methods[f'random64_{seed}']=[reads@(reads.T@q@reads)@reads.T for q in qs]
    frozen={'config':cfg,'selected_weight':chosen,'selection':selection}
    STEM.with_name(STEM.name+'_frozen.json').write_text(json.dumps(frozen,indent=2))
    arrays={'mean':mean.numpy(),'old_inputs':old.numpy(),'axes':axes.cpu().numpy(),'exact_forms':torch.stack(qs).numpy(),'role_secant32':torch.stack(role).numpy()}
    arrays.update({name:torch.stack(forms).numpy() for name,forms in methods.items()})
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**arrays)
    print('frozen before test',chosen,flush=True)
    test,tcoef,treplay=collect(model,corpus,cfg['test_start'],cfg,axes,test=True)
    groups={'fit_heads':[f'drop{h}' for h in cfg['fit_heads']], 'heldout_heads':[f'drop{h}' for h in cfg['heldout_heads']], 'half':[f'half{h}' for h in range(model.config.n_head)], 'pairs':[f'pair{a}_{b}' for a,b in cfg['pair_removals']]}
    results={}
    for name,forms in methods.items():
        results[name]=evaluate(test,tcoef,forms,qs,mean,old,groups)
        print(name,results[name],flush=True)
    results['role_secant32']=evaluate(test,tcoef,role,qs,mean,old,groups,secant=True)
    results['exact']=evaluate(test,tcoef,qs,qs,mean,old,groups)
    # A response rule need not integrate to a scalar. Test a finite triangle.
    x=test['native'];y=test['drop1'];z=test['drop3']
    cycle=[];reference=[]
    for q,a in zip(qs,role):
        cycle.append(response(x,y,q,a,mean,True)+response(y,z,q,a,mean,True)+response(z,x,q,a,mean,True))
        reference.append(quadratic(y,q)-quadratic(x,q))
    cycle_ratio=float(torch.stack(cycle).square().sum().sqrt()/torch.stack(reference).square().sum().sqrt())
    natural=results['natural64'];mixed=results['mixed64'];sym=results['symmetric_role32']
    gates={'exact':max(dreplay,treplay)<=1e-5 and min(results['exact']['groups'].values())>=1-1e-8,
           'heldout_heads':mixed['groups']['heldout_heads']>=.8 and mixed['groups']['heldout_heads']-natural['groups']['heldout_heads']>=.1,
           'heldout_families':all(mixed['groups'][k]-natural['groups'][k]>=.1 for k in ('half','pairs')),
           'natural_retention':mixed['natural_r2']>=natural['natural_r2']-.05,
           'role_function':sym['groups']['heldout_heads']>=.8 and sym['natural_r2']>=natural['natural_r2']-.05}
    out={**frozen,'gates':gates,'head_replay_rms':[dreplay,treplay],'role_cycle_relative_rms':cycle_ratio,'results':results}
    STEM.with_suffix('.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    cache={f'input_{k}':v.float().numpy() for k,v in test.items()};cache.update({f'coefficient_{k}':v.numpy() for k,v in tcoef.items()})
    np.savez_compressed(STEM.with_name(STEM.name+'_test.npz'),**cache)
    lines=['# L8 intervention-aware input metrics','','Four scalar output programs; every state decoder uses at most 64 quadratic reads plus one exact linear read per scalar. The role secant is only a response diagnostic.', '',f'Selected mixture weight: {chosen}. Locked gates: `{gates}`.','','| Method | Natural R2 | Fit-head removal | Held-out heads | Half removal | Paired removal |','|---|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        lines.append(f"| {name} | {r['natural_r2']:.3f} | {r['groups']['fit_heads']:.3f} | {r['groups']['heldout_heads']:.3f} | {r['groups']['half']:.3f} | {r['groups']['pairs']:.3f} |")
    lines += ['',f'Unsymmetrized role-secant triangle-cycle relative RMS: {cycle_ratio:.3f}. Nonzero cycles rule out interpreting that response rule as the differences of a single state function. Its natural R2 column evaluates the symmetrized scalar form, not an independently valid secant-based state decoder.','','The mixture weight was selected on discovery rows using even-head removal only, then frozen before test collection. Odd heads, half removals, and paired removals are held-out perturbations. This is local source-response fidelity for four output directions, not whole-layer reconstruction or semantic identification.']
    STEM.with_name('BILIN18_L8_INTERVENTION_INPUT_METRIC.md').write_text('\n'.join(lines)+'\n')
    print(gates,flush=True)


if __name__=='__main__':main()
