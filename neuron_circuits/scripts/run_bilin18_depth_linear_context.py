#!/usr/bin/env python3
"""Shared linear context versus retained quadratic interactions across depth."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_source_support_factorial import collect_sources
from run_bilin18_l8_context_role_compression import B,response_rules,state_rules
from run_bilin18_l8_execution_alignment import angle_matched_payload
ROOT=Path(__file__).resolve().parents[1];STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_depth_linear_context'

def fit_ridge(delta,response,strength):
    gram=delta.T@delta;lam=strength*torch.trace(gram)/gram.shape[0]
    return torch.linalg.solve(gram+lam*torch.eye(len(gram),device=gram.device,dtype=gram.dtype),delta.T@response)

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());torch.manual_seed(cfg['seed']);model=load_full_model('cuda');model.embedding.float();model.blocks.float()
    rows={};results={};heads={};natural={};audits={};frozen={}
    for layer in cfg['layers']:
        block=model.blocks[layer];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]]
        dc,dh,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'layer':layer});dx=F.rms_norm(dc,(1152,));mu=dx.mean(0);z=dx-mu;_,basis=torch.linalg.eigh(z.T@z);basis=basis.flip(1);native_d=B(dx,dx,L,R,D);native_mean=native_d.mean(0)
        maps={r:basis[:,:r].T for r in cfg['context_ranks']};deltas=[];responses=[]
        for h in range(9):
            y=F.rms_norm(dc-dh[:,h],(1152,));deltas.append(y-dx);responses.append(B(y,y,L,R,D)-native_d)
        delta_d=torch.cat(deltas);response_d=torch.cat(responses);ridge={f'ridge{s:g}':fit_ridge(delta_d.double(),response_d.double(),s).float() for s in cfg['ridge_strengths']}
        def states(x):
            pred={'mean_linear':2*B(mu[None],x-mu,L,R,D),**{n:(x-mu)@w for n,w in ridge.items()}}
            for r,e in maps.items():pred[f'cross{r}']=state_rules(x,mu,e,e.T,L,R,D)['cross']
            return pred
        dstate=states(dx);calibration={n:native_mean-v.mean(0) for n,v in dstate.items()}
        frozen[f'L{layer}_mean']=mu.cpu().numpy();frozen[f'L{layer}_pca']=basis[:,:128].cpu().numpy()
        for n,w in ridge.items():frozen[f'L{layer}_{n}']=w.cpu().numpy()
        pre,hw,replay=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),{**cfg,'layer':layer});x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);den=(native-native_mean).square().sum()
        natural[str(layer)]={n:float(1-(v+calibration[n]-native).square().sum()/den) for n,v in states(x).items()};audits[str(layer)]={'head_replay':replay,'finite_identity_error':0.,'integrable_difference_error':0.}
        natural_scale=(native-native_mean).square().sum(-1).mean()
        for h in range(9):
            ref=hw[:,h];source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;gen=torch.Generator(device='cuda').manual_seed(cfg['seed']+100*layer+h);noises=[torch.randn(len(pre),128,generator=gen,device='cuda') for _ in range(cfg['sham_draws'])]
            for base in cfg['baselines']:
                raw=pre if base=='intact' else pre-ref;sign=-1 if base=='intact' else 1;xn=F.rms_norm(raw,(1152,));fx=B(xn,xn,L,R,D);payloads={'actual':[ref],'angle':[angle_matched_payload(raw,ref,source,n) for n in noises]}
                for family,draws in payloads.items():
                    for scale in cfg['scales']:
                        prefix=f'L{layer}_{base}_{family}_s{scale:g}';acc={};headenergy=0.
                        for p in draws:
                            yn=F.rms_norm(raw+sign*scale*p,(1152,));delta=yn-xn;target=B(yn,yn,L,R,D)-fx;energy=target.square().sum(-1);valid=energy>1e-20;headenergy+=float(energy.sum())
                            exact=2*B((xn+yn)/2,delta,L,R,D);audits[str(layer)]['finite_identity_error']=max(audits[str(layer)]['finite_identity_error'],float((exact-target).norm()/target.norm()))
                            pred={'mean_linear':2*B(mu[None],delta,L,R,D),**{n:delta@w for n,w in ridge.items()}}
                            for r,e in maps.items():
                                rules=response_rules(xn,yn,mu,e,e.T,L,R,D);pred[f'context{r}']=rules['context_only'];pred[f'cross{r}']=rules['cross']
                            if h==0 and family=='actual' and scale==1:
                                sx,sy=states(xn),states(yn)
                                for n in sx:audits[str(layer)]['integrable_difference_error']=max(audits[str(layer)]['integrable_difference_error'],float((sy[n]-sx[n]-pred[n]).norm()/target.norm()))
                            for label,v in [('energy',energy),('count',valid.float())]:
                                key=prefix+'_'+label;rows.setdefault(key,np.zeros(cfg['rows']));rows[key]+=v.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
                            for n,v in pred.items():
                                err=(v-target).square().sum(-1);ratio=torch.where(valid,err/energy.clamp_min(1e-20),torch.zeros_like(energy));acc[n]=acc.get(n,0.)+float(err.sum())
                                for label,t in [('error',err),('site_ratio',ratio)]:
                                    key=prefix+'_'+n+'_'+label;rows.setdefault(key,np.zeros(cfg['rows']));rows[key]+=t.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
                        for n,e in acc.items():heads[prefix+'_'+n+f'_h{h}']=1-e/headenergy
            print('layer',layer,'head',h,'complete',flush=True)
        methods=['mean_linear']+list(ridge)+[f'{m}{r}' for r in maps for m in ['context','cross']]
        for base in cfg['baselines']:
            for family in ['actual','angle']:
                for scale in cfg['scales']:
                    prefix=f'L{layer}_{base}_{family}_s{scale:g}'
                    for n in methods:
                        key=prefix+'_'+n;results[key]={'pooled':float(1-rows[key+'_error'].sum()/rows[prefix+'_energy'].sum()),'site_mean':float(1-rows[key+'_site_ratio'].sum()/rows[prefix+'_count'].sum()),'head_macro':float(np.mean([heads[key+f'_h{h}'] for h in range(9)])),'mse_over_natural_variance':float(rows[key+'_error'].sum()/rows[prefix+'_count'].sum()/natural_scale)}
        STEM.with_name(STEM.name+'_partial.json').write_text(json.dumps({'completed_layer':layer,'results':results,'natural':natural}))
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**frozen)
    rng=np.random.default_rng(cfg['seed']);ix=rng.integers(cfg['rows'],size=(2000,cfg['rows']))
    for key,v in results.items():
        prefix='_'.join(key.split('_')[:4]);e=rows[key+'_error'];den=rows[prefix+'_energy'];v['document_bootstrap95']=np.quantile(1-e[ix].sum(1)/den[ix].sum(1),[.025,.975]).tolist()
    def allconditions(method):return all(results[f'L17_{b}_{f}_s{s:g}_{method}']['pooled']>=.9 for b in cfg['baselines'] for f in ['actual','angle'] for s in cfg['scales'])
    gates={'late_mean_linear':allconditions('mean_linear'),'late_learned_linear':any(allconditions(f'ridge{s:g}') and natural['17'][f'ridge{s:g}']>=.9 for s in cfg['ridge_strengths']),'late_compact_quadratic':allconditions('cross16') and natural['17']['cross16']>=.9,'middle_mean_insufficient':results['L8_intact_actual_s1_mean_linear']['pooled']<=.5}
    output={'config':cfg,'gates':gates,'results':results,'natural_r2':natural,'head_results':heads,'audits':audits};STEM.with_suffix('.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    lines=['# Shared linear context versus quadratic context computation','','| Layer | Mean-linear | Ridge .001 | Ridge .01 | Ridge .1 | Cross16 | Cross64 | Cross128 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for l in cfg['layers']:lines.append(f'| L{l} | '+' | '.join(f"{results[f'L{l}_intact_actual_s1_{n}']['pooled']:.3f}" for n in ['mean_linear','ridge0.001','ridge0.01','ridge0.1','cross16','cross64','cross128'])+' |')
    lines+=['',f'Locked gates: {gates}.','',cfg['scope']];STEM.with_name(STEM.name.upper()+'.md').write_text('\n'.join(lines)+'\n');print(gates,flush=True)
if __name__=='__main__':main()
