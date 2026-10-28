#!/usr/bin/env python3
"""Discriminate source geometry, payload semantics, normalization, and edit scale."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_qkov_causal_edges import load_full_model,attention_details
from run_bilin18_l8_intervention_input_metric import covariance,metric_form
from run_bilin18_l8_source_span_response import source_form

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_source_support_factorial'


def norm_match(candidate,reference):
    return candidate*reference.norm(dim=-1,keepdim=True)/candidate.norm(dim=-1,keepdim=True).clamp_min(1e-30)


@torch.inference_mode()
def collect_sources(model,corpus,cfg):
    dim=model.config.n_embd;heads=model.config.n_head;hd=dim//heads
    pres=[];writes=[];error=0.;energy=0.
    for offset in range(0,cfg['rows'],cfg['batch_size']):
        ids=torch.as_tensor(np.array(corpus[cfg['start']+offset:cfg['start']+offset+cfg['batch_size'],:128],dtype=np.int64),device='cuda')
        x0=F.rms_norm(model.embedding(ids),(dim,));x=x0;bus=None
        for layer,block in enumerate(model.blocks):
            mixed=block.lambdas[0]*x+block.lambdas[1]*x0
            detail=attention_details(block.attn,F.rms_norm(mixed,(dim,)),bus,model.config,model.tt);bus=detail['value_bus']
            pre=mixed+detail['write']
            if layer==cfg.get("layer",8):
                p=cfg['positions'];attended=torch.einsum('bhpk,bkhd->bphd',detail['pattern'][:,:,p],detail['mixed_value'])
                hw=torch.einsum('bphd,ohd->bpho',attended,block.attn.c_proj.weight.reshape(dim,heads,hd))
                error+=float((hw.sum(2)-detail['write'][:,p]).square().sum());energy+=float(detail['write'][:,p].square().sum())
                pres.append(pre[:,p].flatten(0,1));writes.append(hw.flatten(0,1));break
            normalized=F.rms_norm(pre,(dim,));x=pre+block.mlp.Down(block.mlp.Left(normalized)*block.mlp.Right(normalized))+block.mlp.Down_bias
        if (offset+cfg['batch_size'])%32==0:print(f'factorial collection {offset+cfg["batch_size"]}',flush=True)
    return torch.cat(pres),torch.cat(writes),float(np.sqrt(error/energy))


def changes(x,changed,forms,exact,mean):
    delta=changed-x;mid=(changed+x)/2-mean
    linear=2*torch.einsum('nd,adj,j->na',delta,exact,mean)
    return linear+2*torch.einsum('nd,adj,nj->na',delta,forms,mid)


@torch.inference_mode()
def main():
    torch.set_num_threads(6)
    cfg=json.loads((ROOT/'configs/bilin18_l8_source_support_factorial.json').read_text())
    print('factorial setup',flush=True)
    torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    frozen=np.load(ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_intervention_input_metric_frozen.npz')
    charts=np.load(ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_hybrid_source_response_frozen.npz')
    old=torch.from_numpy(frozen['old_inputs']);mean=torch.from_numpy(frozen['mean']);qs=list(torch.from_numpy(frozen['exact_forms']));cx=covariance(old-mean)
    natural128=[metric_form(q,cx,128) for q in qs];natural256=[metric_form(q,cx,256) for q in qs]
    print('factorial natural forms ready',flush=True)
    forms={}
    for h in cfg['heads']:
        forms[h]={'natural256':torch.stack(natural256).cuda()}
        for kind in ('correct','wrong'):
            basis=torch.from_numpy(charts[f'rank64_{h}_{kind}'])
            forms[h][f'hybrid256_{kind}']=torch.stack([a+source_form(q-a,basis) for q,a in zip(qs,natural128)]).cuda()
    mean=mean.cuda();exact=torch.stack(qs).cuda();axes=torch.from_numpy(frozen['axes']).cuda().double()
    print('factorial source forms ready',flush=True)
    model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[8]
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');pre,heads,replay=collect_sources(model,corpus,cfg)
    dim=pre.shape[-1];xfloat=F.rms_norm(pre,(dim,));x=xfloat.double()
    rms=(pre.square().mean(-1,keepdim=True)+torch.finfo(pre.dtype).eps).sqrt()
    native=torch.einsum('nd,adj,nj->na',x,exact,x)
    train=torch.from_numpy(frozen['old_inputs']).cuda();train_mean=torch.einsum('nd,adj,nj->na',train,exact,train).mean(0)
    natural_variance=float((native-train_mean).square().sum())*len(cfg['heads'])
    methods=list(forms[cfg['heads'][0]])
    accum={};norm_error=0.;audit_error=0.;audit_energy=0.
    for h in cfg['heads']:
        actual=heads[:,h];permutation=torch.as_tensor(rng.permutation(len(pre)),device='cuda')
        w=block.attn.c_proj.weight.reshape(dim,model.config.n_head,-1)[:,h]
        payloads={'actual':actual,'shuffled_same_head':norm_match(actual[permutation],actual),
                  'gaussian_same_head':norm_match(torch.randn(len(pre),w.shape[1],device='cuda')@w.T,actual),
                  'isotropic_residual':norm_match(torch.randn_like(actual),actual)}
        for payload,perturbation in payloads.items():
            norm_error=max(norm_error,float((perturbation.norm(dim=-1)-actual.norm(dim=-1)).abs().max()/actual.norm(dim=-1).max()))
            for normalization in cfg['normalizers']:
                for scale in cfg['scales']:
                    changed_float=F.rms_norm(pre-scale*perturbation,(dim,)) if normalization=='recompute' else xfloat-scale*perturbation/rms
                    changed=changed_float.double()
                    target=changes(x,changed,exact,exact,mean)
                    if scale==1. and normalization=='recompute':
                        def native_coefficient(z):return block.mlp.Down(block.mlp.Left(z)*block.mlp.Right(z)).double()@axes.T
                        direct=native_coefficient(changed_float[:16])-native_coefficient(xfloat[:16])
                        audit_error+=float((target[:16]-direct).square().sum());audit_energy+=float(direct.square().sum())
                    key=f'{payload}|{normalization}|{scale:g}'
                    if key not in accum:accum[key]={'energy':np.zeros(cfg['rows']),'errors':{m:np.zeros(cfg['rows']) for m in methods}}
                    row=accum[key];row['energy']+=target.square().sum(-1).reshape(cfg['rows'],len(cfg['positions'])).sum(1).cpu().numpy()
                    for method,a in forms[h].items():
                        prediction=changes(x,changed,a,exact,mean)
                        row['errors'][method]+=(prediction-target).square().sum(-1).reshape(cfg['rows'],len(cfg['positions'])).sum(1).cpu().numpy()
        print(f'factorial head {h} complete',flush=True)
    indices=rng.integers(cfg['rows'],size=(cfg['bootstrap_replicates'],cfg['rows']))
    results={}
    for key,row in accum.items():
        energy=row['energy'];results[key]={'effect_rms_over_natural_sd':float(np.sqrt(energy.sum()/natural_variance)),'methods':{}}
        boots={}
        for name,error in row['errors'].items():
            boots[name]=1-error[indices].sum(1)/energy[indices].sum(1)
            results[key]['methods'][name]={'fidelity':float(1-error.sum()/energy.sum()),'mse_over_natural_variance':float(error.sum()/natural_variance),'row_bootstrap_95':np.quantile(boots[name],[.025,.975]).tolist()}
        results[key]['hybrid_advantage_95']=np.quantile(boots['hybrid256_correct']-boots['natural256'],[.025,.975]).tolist()
    def row(payload='actual',norm='recompute',scale=1.):return results[f'{payload}|{norm}|{scale:g}']
    def gain(payload):
        m=row(payload)['methods'];return m['hybrid256_correct']['fidelity']-m['natural256']['fidelity']
    full=row()['methods']['natural256'];freeze=row(norm='frozen')['methods']['natural256'];small=row(scale=.05)['methods']['natural256']
    numerical=float(np.sqrt(audit_error/audit_energy))
    gates={'same_support_benefit':all(gain(p)>=.05 and gain(p)>=gain('actual')-.05 for p in ('shuffled_same_head','gaussian_same_head')),
           'support_specificity':gain('isotropic_residual')<=.03 and gain('actual')-gain('isotropic_residual')>=.05,
           'normalization_primary':freeze['fidelity']-full['fidelity']>=.1 and freeze['mse_over_natural_variance']<=.5*full['mse_over_natural_variance'],
           'small_edit_recovery':small['fidelity']>=.9 and small['fidelity']-full['fidelity']>=.1,
           'exact':numerical<=1e-4 and norm_error<=1e-5 and replay<=1e-5}
    STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'gates':gates,'head_replay_rms':replay,'native_change_replay_rms':numerical,'norm_matching_relative_error':norm_error,'results':results},indent=2,allow_nan=False))
    np.savez_compressed(STEM.with_name(STEM.name+'_row_errors.npz'),**{f'{key}_energy':r['energy'] for key,r in accum.items()},**{f'{key}_{name}_error':error for key,r in accum.items() for name,error in r['errors'].items()})
    lines=['# Source-support, normalization, and scale factorial','',cfg['scope'],'',f'Locked gates: `{gates}`. Native-change replay relative RMS: `{numerical:.3g}`.','','## Scale 1 controls','','| Payload | RMS | Effect RMS / natural SD | Global256 fidelity | Correct hybrid | Wrong hybrid | Global normalized MSE |','|---|---|---:|---:|---:|---:|---:|']
    for p in cfg['payloads']:
        for n in cfg['normalizers']:
            r=row(p,n);m=r['methods']
            lines.append(f"| {p} | {n} | {r['effect_rms_over_natural_sd']:.3f} | {m['natural256']['fidelity']:.3f} | {m['hybrid256_correct']['fidelity']:.3f} | {m['hybrid256_wrong']['fidelity']:.3f} | {m['natural256']['mse_over_natural_variance']:.6f} |")
    lines+=['','## Actual-head magnitude curve with recomputed RMS','','| Scale | Global256 | Correct hybrid | Wrong hybrid |','|---|---:|---:|---:|']
    for scale in cfg['scales']:
        m=row(scale=scale)['methods'];lines.append(f"| {scale} | {m['natural256']['fidelity']:.3f} | {m['hybrid256_correct']['fidelity']:.3f} | {m['hybrid256_wrong']['fidelity']:.3f} |")
    lines+=['','Same-head shams are norm matched but not semantically matched. Frozen normalization is an artificial local counterfactual. Bootstrap intervals in JSON resample complete text rows; no result establishes model-wide semantic coverage. Small absolute errors relative to natural variation do not imply behavioral irrelevance.']
    STEM.with_name('BILIN18_L8_SOURCE_SUPPORT_FACTORIAL.md').write_text('\n'.join(lines)+'\n')
    print(gates,flush=True)


if __name__=='__main__':main()
