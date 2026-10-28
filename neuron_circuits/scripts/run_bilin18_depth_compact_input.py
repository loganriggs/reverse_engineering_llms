#!/usr/bin/env python3
"""Distinguish a compact input code from a compact context with broad partners."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_source_support_factorial import collect_sources
from run_bilin18_l8_context_role_compression import B
from run_bilin18_l8_execution_alignment import angle_matched_payload
ROOT=Path(__file__).resolve().parents[1];STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_depth_compact_input'
def compact_response(x,y,mu,basis,L,R,D):
    delta=y-x;mid=(x+y)/2-mu;pd=(delta@basis)@basis.T;pm=(mid@basis)@basis.T
    quadratic=2*B(pm,pd,L,R,D)
    return {'strict':2*B(mu[None],pd,L,R,D)+quadratic,'both_full_linear':2*B(mu[None],delta,L,R,D)+quadratic}
@torch.inference_mode()
def main():
    torch.set_num_threads(6);prior=json.loads((STEM.parent/'bilin18_depth_linear_context.json').read_text());cfg=prior['config'];spec=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());frozen=np.load(STEM.parent/'bilin18_depth_linear_context_frozen.npz');model=load_full_model('cuda');model.embedding.float();model.blocks.float();rows={};results={};natural={};replay=0.
    for layer in cfg['layers']:
        L,R,D=[m.weight for m in [model.blocks[layer].mlp.Left,model.blocks[layer].mlp.Right,model.blocks[layer].mlp.Down]];mu=torch.from_numpy(frozen[f'L{layer}_mean']).cuda();basis=torch.from_numpy(frozen[f'L{layer}_pca']).cuda()
        dc,_,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'layer':layer});dx=F.rms_norm(dc,(1152,));native_d=B(dx,dx,L,R,D);native_mean=native_d.mean(0)
        pre,hw,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),{**cfg,'layer':layer});xn=F.rms_norm(pre,(1152,));native=B(xn,xn,L,R,D);natural[str(layer)]={}
        for r in cfg['context_ranks']:
            q=basis[:,:r]
            def state(x):
                z=x-mu;p=(z@q)@q.T
                return {'strict':B(mu+p,mu+p,L,R,D),'both_full_linear':2*B(mu[None],z,L,R,D)+B(p,p,L,R,D)}
            ds=state(dx);ts=state(xn)
            for n in ds:natural[str(layer)][f'{n}{r}']=float(1-(ts[n]+native_mean-ds[n].mean(0)-native).square().sum()/(native-native_mean).square().sum())
        for h in range(9):
            ref=hw[:,h];source=torch.linalg.qr(model.blocks[layer].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;gen=torch.Generator(device='cuda').manual_seed(cfg['seed']+100*layer+h);noise=[torch.randn(len(pre),128,generator=gen,device='cuda') for _ in range(cfg['sham_draws'])]
            for base in cfg['baselines']:
                raw=pre if base=='intact' else pre-ref;sign=-1 if base=='intact' else 1;x=F.rms_norm(raw,(1152,));fx=B(x,x,L,R,D);payloads={'actual':[ref],'angle':[angle_matched_payload(raw,ref,source,n) for n in noise]}
                for fam,draws in payloads.items():
                    for scale in cfg['scales']:
                        prefix=f'L{layer}_{base}_{fam}_s{scale:g}'
                        for p in draws:
                            y=F.rms_norm(raw+sign*scale*p,(1152,));target=B(y,y,L,R,D)-fx;energy=target.square().sum(-1);valid=energy>1e-20
                            for label,v in [('energy',energy),('count',valid.float())]:
                                key=prefix+'_'+label;rows.setdefault(key,np.zeros(cfg['rows']));rows[key]+=v.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
                            for r in cfg['context_ranks']:
                                pred=compact_response(x,y,mu,basis[:,:r],L,R,D)
                                for n,v in pred.items():
                                    error=(v-target).square().sum(-1);ratio=torch.where(valid,error/energy.clamp_min(1e-20),torch.zeros_like(energy))
                                    for label,val in [('error',error),('site_ratio',ratio)]:
                                        key=prefix+f'_{n}{r}_'+label;rows.setdefault(key,np.zeros(cfg['rows']));rows[key]+=val.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
            print(layer,h,'complete',flush=True)
    rng=np.random.default_rng(cfg['seed']);ix=rng.integers(cfg['rows'],size=(2000,cfg['rows']))
    for key,e in rows.items():
        if not key.endswith('_error'):continue
        name=key[:-6];prefix='_'.join(name.split('_')[:4]);den=rows[prefix+'_energy'];results[name]={'pooled':float(1-e.sum()/den.sum()),'site_mean':float(1-rows[name+'_site_ratio'].sum()/rows[prefix+'_count'].sum()),'document_bootstrap95':np.quantile(1-e[ix].sum(1)/den[ix].sum(1),[.025,.975]).tolist()}
    act=natural['17']['strict16']>=.9 and all(results[f'L17_{b}_actual_s{s:g}_strict16']['pooled']>=.9 for b in cfg['baselines'] for s in cfg['scales'])
    gates={'late_actual_code16':act,'late_universal_code16':act and all(results[f'L17_{b}_angle_s{s:g}_strict16']['pooled']>=.9 for b in cfg['baselines'] for s in cfg['scales']),'broad_partner_needed':prior['results']['L17_intact_actual_s1_cross16']['pooled']-results['L17_intact_actual_s1_strict16']['pooled']>=.1}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':spec,'parent_config':cfg,'gates':gates,'natural_r2':natural,'results':results},indent=2))
    lines=['# Compact input coordinates versus broad interaction partners','','| Layer | Strict16 actual | Both16 + full linear actual | Strict16 natural R2 | Strict64 actual | Strict128 actual |','|---|---:|---:|---:|---:|---:|']
    for l in cfg['layers']:
        vals=[results[f'L{l}_intact_actual_s1_{n}']['pooled'] for n in ['strict16','both_full_linear16']]+[natural[str(l)]['strict16']]+[results[f'L{l}_intact_actual_s1_strict{r}']['pooled'] for r in [64,128]]
        lines.append(f'| L{l} | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    lines+=['',f'Locked gates: {gates}.','',spec['scope']];STEM.with_name(STEM.name.upper()+'.md').write_text('\n'.join(lines)+'\n');print(gates)
if __name__=='__main__':main()
