#!/usr/bin/env python3
"""Matched full-output source geometry across depth, with held-out global bases."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_source_support_factorial import collect_sources
from run_bilin18_l8_context_response_geometry import source_jacobian,project
from run_bilin18_l8_context_role_compression import B
from run_bilin18_l8_execution_alignment import angle_matched_payload
ROOT=Path(__file__).resolve().parents[1];STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_depth_source_geometry'

def fit_response_bases(response,rank):
    mean=response.mean(0);centered=response-mean
    def eigenbasis(x):return torch.linalg.eigh(x.T@x).eigenvectors[:,-rank:].flip(1).T
    return {'global':eigenbasis(response),'centered_global':eigenbasis(centered),'mean':(mean/mean.norm().clamp_min(1e-30))[None]},mean

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());torch.manual_seed(cfg['seed'])
    test_manifest=json.loads((ROOT/cfg['corpus']).with_suffix('.manifest.json').read_text());discovery_manifest=json.loads((ROOT/cfg['discovery']).with_suffix('.manifest.json').read_text())
    assert not {v['document_token_sha256'] for v in test_manifest['records']} & {v['document_token_sha256'] for v in discovery_manifest['records']}
    model=load_full_model('cuda');model.embedding.float();model.blocks.float();rows={};heads={};frozen={};audits={};results={};response_means={}
    for layer in cfg['layers']:
        localcfg={**cfg,'layer':layer};block=model.blocks[layer];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]]
        pre_d,hw_d,replay_d=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),localcfg)
        xd=F.rms_norm(pre_d,(1152,));fxd=B(xd,xd,L,R,D);responses=[]
        for h in range(9):
            yd=F.rms_norm(pre_d-hw_d[:,h],(1152,));responses.append(B(yd,yd,L,R,D)-fxd)
        bases,mean=fit_response_bases(torch.cat(responses),64);response_means[str(layer)]=float(mean.square().sum()/torch.cat(responses).square().sum(-1).mean())
        for n,b in bases.items():frozen[f'L{layer}_{n}']=b.cpu().numpy()
        frozen[f'L{layer}_mean_vector']=mean.cpu().numpy()
        pre,hw,replay=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),localcfg);N=len(pre);wrong=(torch.arange(N,device='cuda')+len(cfg['positions']))%N
        audits[str(layer)]={'head_replay_discovery':replay_d,'head_replay_test':replay,'chart_orthogonality':0.,'secant_reversal':0.,'angle_dot_error':0.}
        for h in range(9):
            source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;ref=hw[:,h];secants={}
            gen=torch.Generator(device='cuda').manual_seed(cfg['seed']+100*layer+h);noise=[torch.randn(N,128,generator=gen,device='cuda') for _ in range(cfg['sham_draws'])]
            for base in cfg['baselines']:
                z=pre if base=='intact' else pre-ref;sign=-1 if base=='intact' else 1;x=F.rms_norm(z,(1152,));fx=B(x,x,L,R,D);Js=[];charts=[]
                for off in range(0,N,cfg['matrix_batch']):
                    J=source_jacobian(z[off:off+cfg['matrix_batch']],L,R,D,source);u,_,_=torch.linalg.svd(J,full_matrices=False);u=torch.linalg.qr(u[:,:,:64]).Q;Js.append(J);charts.append(u.transpose(1,2))
                    audits[str(layer)]['chart_orthogonality']=max(audits[str(layer)]['chart_orthogonality'],float((u.transpose(1,2)@u-torch.eye(64,device='cuda')).abs().max()))
                J=torch.cat(Js);chart=torch.cat(charts);allbases={**bases,'matched':chart,'wrong_context':chart[wrong]}
                payloads={'actual':[ref],'angle':[angle_matched_payload(z,ref,source,n) for n in noise]}
                for p in payloads['angle']:audits[str(layer)]['angle_dot_error']=max(audits[str(layer)]['angle_dot_error'],float(((z*p).sum(-1)-(z*ref).sum(-1)).abs().div(z.norm(dim=-1)*ref.norm(dim=-1)).max()))
                for family,draws in payloads.items():
                    for regime in cfg['regimes']:
                        head_energy=0.;head_captured={}
                        for draw,p in enumerate(draws):
                            if regime=='tangent':target=sign*torch.einsum('ndr,nr->nd',J,p@source)
                            else:
                                y=F.rms_norm(z+sign*p,(1152,));target=B(y,y,L,R,D)-fx
                            if family=='actual' and regime=='finite':secants[base]=target
                            energy=target.square().sum(-1);valid=energy>1e-20;prefix=f'L{layer}_{base}_{family}_{regime}';head_energy+=float(energy.sum())
                            for label,v in [('energy',energy),('count',valid.float())]:
                                key=prefix+'_'+label;rows.setdefault(key,np.zeros(cfg['rows']));rows[key]+=v.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
                            for chartname,basis in allbases.items():
                                ranks=[1] if chartname=='mean' else cfg['ranks']
                                for rank in ranks:
                                    selected=basis[:rank] if basis.ndim==2 else basis[:,:rank];cap=project(target,selected).square().sum(-1);ratio=torch.where(valid,cap/energy.clamp_min(1e-20),torch.zeros_like(energy));suffix=f'{chartname}_r{rank}'
                                    head_captured[suffix]=head_captured.get(suffix,0.)+float(cap.sum())
                                    for label,v in [('captured',cap),('site_ratio',ratio)]:
                                        key=prefix+'_'+suffix+'_'+label;rows.setdefault(key,np.zeros(cfg['rows']));rows[key]+=v.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
                        for suffix,cap in head_captured.items():heads[prefix+'_'+suffix+f'_h{h}']=cap/head_energy
            audits[str(layer)]['secant_reversal']=max(audits[str(layer)]['secant_reversal'],float((secants['intact']+secants['removed']).norm()/secants['intact'].norm()));print('layer',layer,'head',h,'complete',flush=True)
        STEM.with_name(STEM.name+'_partial.json').write_text(json.dumps({'completed_layer':layer,'head_results':heads,'audits':audits}))
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**frozen)
    for key in rows:
        if not key.endswith('_captured'):continue
        short=key[:-9];prefix=short.split('_global_')[0] if '_global_' in short and '_centered_global_' not in short else None
        # Prefix consists of layer, baseline, family and regime, all single words.
        prefix='_'.join(short.split('_')[:4]);results[short]={'pooled':float(rows[key].sum()/rows[prefix+'_energy'].sum()),'site_mean':float(rows[short+'_site_ratio'].sum()/rows[prefix+'_count'].sum()),'head_macro':float(np.mean([heads[short+f'_h{h}'] for h in range(9)]))}
    rng=np.random.default_rng(cfg['seed']);ix=rng.integers(cfg['rows'],size=(2000,cfg['rows']));contrasts={}
    def value(l,b,f,g,c,r=16,m='pooled'):return results[f'L{l}_{b}_{f}_{g}_{c}_r{r}'][m]
    def boot(l,b,f,g,c,m):
        prefix=f'L{l}_{b}_{f}_{g}';key=prefix+f'_{c}_r16';num=rows[key+('_captured' if m=='pooled' else '_site_ratio')];den=rows[prefix+('_energy' if m=='pooled' else '_count')];return num[ix].sum(1)/den[ix].sum(1)
    for layer in cfg['layers']:
        for base in cfg['baselines']:
            for g in cfg['regimes']:
                for m in ['pooled','site_mean']:
                    gap=boot(layer,base,'actual',g,'matched',m)-boot(layer,base,'angle',g,'matched',m)
                    contrasts[f'L{layer}_{base}_{g}_{m}']={'actual_minus_angle':value(layer,base,'actual',g,'matched',m=m)-value(layer,base,'angle',g,'matched',m=m),'document_bootstrap95':np.quantile(gap,[.025,.975]).tolist()}
    for layer in [1,17]:
        for m in ['pooled','site_mean']:
            diff=boot(layer,'intact','actual','finite','global',m)-boot(8,'intact','actual','finite','global',m)
            contrasts[f'L{layer}_minus_L8_global_finite_{m}']={'value':value(layer,'intact','actual','finite','global',m=m)-value(8,'intact','actual','finite','global',m=m),'document_bootstrap95':np.quantile(diff,[.025,.975]).tolist()}
    gates={'late_shared_compactness':all(value(17,b,'actual','finite','global')>=.8 and value(17,b,'actual','finite','global')-value(8,b,'actual','finite','global')>=.2 for b in cfg['baselines']),
      'early_shared_advantage':value(1,'intact','actual','finite','global')-value(8,'intact','actual','finite','global')>=.1,
      'late_context_invariance':all(value(17,b,'actual','finite','matched')-value(17,b,'actual','finite','wrong_context')<=.05 and value(8,b,'actual','finite','matched')-value(8,b,'actual','finite','wrong_context')>=.1 for b in cfg['baselines']),
      'alignment_all_depths':all(contrasts[f'L{l}_removed_{g}_pooled']['actual_minus_angle']>=.08 for l in cfg['layers'] for g in cfg['regimes'])}
    output={'config':cfg,'gates':gates,'results':results,'head_results':heads,'contrasts':contrasts,'audits':audits,'discovery_response_mean_energy_fraction':response_means}
    STEM.with_suffix('.json').write_text(json.dumps(output,indent=2,allow_nan=False));lines=['# Matched source-response geometry across depth','','| Layer / baseline | Global16 actual | Matched16 actual | Wrong-context16 actual | Matched16 angle | Global4 actual | Mean-direction1 actual |','|---|---:|---:|---:|---:|---:|---:|']
    for l in cfg['layers']:
        for b in cfg['baselines']:
            vals=[value(l,b,'actual','finite',c) for c in ['global','matched','wrong_context']]+[value(l,b,'angle','finite','matched'),value(l,b,'actual','finite','global',4),value(l,b,'actual','finite','mean',1)]
            lines.append(f'| L{l} / {b} | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    lines+=['',f'Locked gates: {gates}.','',cfg['scope']];STEM.with_name(STEM.name.upper()+'.md').write_text('\n'.join(lines)+'\n');print(gates,flush=True)
if __name__=='__main__':main()
