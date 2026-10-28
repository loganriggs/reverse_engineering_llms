#!/usr/bin/env python3
"""Separate head self-inclusion from alignment with the remaining residual."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_source_support_factorial import collect_sources,norm_match
from run_bilin18_l8_context_response_geometry import source_jacobian,project
from run_bilin18_l8_context_role_compression import B
from run_bilin18_l8_channel_coordination import canonical_weights,permuted_weights
from run_bilin18_l8_execution_alignment import shifted_indices,angle_matched_payload
ROOT=Path(__file__).resolve().parents[1];STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_source_self_inclusion'

def baseline_and_edit(pre,ref,baseline):
    return pre if baseline=='intact' else pre-ref

@torch.inference_mode()
def main():
    global STEM
    if '--document-replication' in sys.argv:STEM=STEM.with_name('bilin18_l8_document_replication')
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());torch.manual_seed(cfg['seed'])
    model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[8]
    weights=canonical_weights(*(m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]))
    frozen=np.load(STEM.parent/'bilin18_l8_canonical_channel_coordination_frozen.npz')
    permutations={r:torch.from_numpy(frozen[f'permutation{r}']).cuda() for r in range(3)}
    cases={'native':('native',0),'joint':('joint',0),**{f'{kind}{r}':(kind,r) for kind in ['right','down','direction'] for r in range(3)}}
    pre,hw,replay=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);N=len(pre)
    sources=[torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
    rng=np.random.default_rng(cfg['seed']);payloads={};audits={};arrays={'pre':pre.cpu().numpy(),'heads':hw.cpu().numpy()}
    for h,source in enumerate(sources):
        ref=hw[:,h];common={'actual':[ref],'shuffled':[],'gaussian':[]};noises=[]
        shifts=rng.choice(np.arange(1,cfg['rows']),size=cfg['sham_draws'],replace=False)
        for draw,shift in enumerate(shifts):
            idx=shifted_indices(cfg['rows'],len(cfg['positions']),int(shift),'cuda')
            common['shuffled'].append(norm_match(ref[idx],ref));gen=torch.Generator(device='cuda').manual_seed(cfg['seed']+100*h+draw)
            common['gaussian'].append(norm_match(torch.randn(N,128,generator=gen,device='cuda')@source.T,ref))
            noises.append(torch.randn(N,128,generator=gen,device='cuda'))
        for base in cfg['baselines']:
            z=baseline_and_edit(pre,ref,base);payloads[h,base]={**common,'angle':[angle_matched_payload(z,ref,source,n) for n in noises]}
            audits[f'h{h}_{base}']={'shifts':shifts.tolist(),'reconstruction_error':float(((z+ref if base=='removed' else z)-pre).norm()/pre.norm()),
              'angle_dot_error':max(float(((z*p).sum(-1)-(z*ref).sum(-1)).abs().div(z.norm(dim=-1)*ref.norm(dim=-1)).max()) for p in payloads[h,base]['angle']),
              'angle_norm_error':max(float((p.norm(dim=-1)-ref.norm(dim=-1)).abs().div(ref.norm(dim=-1)).max()) for p in payloads[h,base]['angle'])}
            for fam,draws in payloads[h,base].items():
                for i,p in enumerate(draws):arrays[f'h{h}_{base}_{fam}{i}']=p.cpu().numpy()
    np.savez_compressed(STEM.with_name(STEM.name+'_payloads.npz'),**arrays);print('Payloads frozen',flush=True)
    rows={};heads={};draw_metrics={};orth_error=0.;secant_error=0.
    for name,(kind,rep) in cases.items():
        L,R,D=permuted_weights(*weights,permutations[rep],kind)
        for h,source in enumerate(sources):
            ref=hw[:,h];secants={}
            for base in cfg['baselines']:
                z=baseline_and_edit(pre,ref,base);x=F.rms_norm(z,(1152,));fx=B(x,x,L,R,D);Js=[];charts=[]
                for off in range(0,N,cfg['matrix_batch']):
                    J=source_jacobian(z[off:off+cfg['matrix_batch']],L,R,D,source);u,_,_=torch.linalg.svd(J,full_matrices=False);u=torch.linalg.qr(u[:,:,:64]).Q
                    Js.append(J);charts.append(u.transpose(1,2));orth_error=max(orth_error,float((u.transpose(1,2)@u-torch.eye(64,device='cuda')).abs().max()))
                J=torch.cat(Js);chart=torch.cat(charts)
                for fam,draws in payloads[h,base].items():
                    acc={g:{'energy':0.,'captured':{r:0. for r in cfg['ranks']}} for g in cfg['regimes']}
                    for draw,p in enumerate(draws):
                        tangent=torch.einsum('ndr,nr->nd',J,p@source)
                        for regime in cfg['regimes']:
                            if regime=='tangent':target=tangent
                            else:
                                scale=(.05 if 'small' in regime else 1.)*(-1 if 'remove' in regime else 1.)
                                y=F.rms_norm(z+scale*p,(1152,));target=B(y,y,L,R,D)-fx
                            if fam=='actual' and ((base=='intact' and regime=='remove') or (base=='removed' and regime=='add')):secants[base]=target
                            energy=target.square().sum(-1);valid=energy>1e-20;prefix=f'{name}_{base}_{fam}_{regime}'
                            acc[regime]['energy']+=float(energy.sum())
                            for label,vals in [('energy',energy),('count',valid.float())]:
                                key=prefix+'_'+label;rows.setdefault(key,np.zeros(cfg['rows']));rows[key]+=vals.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
                            for rank in cfg['ranks']:
                                captured=project(target,chart[:,:rank]).square().sum(-1);ratio=torch.where(valid,captured/energy.clamp_min(1e-20),torch.zeros_like(energy));acc[regime]['captured'][rank]+=float(captured.sum())
                                for label,vals in [('captured',captured),('site_ratio',ratio)]:
                                    key=prefix+f'_r{rank}_'+label;rows.setdefault(key,np.zeros(cfg['rows']));rows[key]+=vals.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
                                draw_metrics[f'{prefix}_h{h}_draw{draw}_r{rank}']={'pooled':float(captured.sum()/energy.sum()),'site_mean':float(ratio.sum()/valid.sum())}
                    for g,v in acc.items():heads[f'{name}_{base}_{fam}_{g}_h{h}']={str(r):v['captured'][r]/v['energy'] for r in cfg['ranks']}
            secant_error=max(secant_error,float((secants['intact']+secants['removed']).norm()/secants['intact'].norm()))
            print(name,'head',h,'complete',flush=True)
        STEM.with_name(STEM.name+'_partial.json').write_text(json.dumps({'completed':name,'head_results':heads}))
    results={};families=['actual','shuffled','gaussian','angle']
    for name in cases:
        for base in cfg['baselines']:
            for fam in families:
                for g in cfg['regimes']:
                    prefix=f'{name}_{base}_{fam}_{g}';results[prefix]={}
                    for rank in cfg['ranks']:
                        results[prefix][str(rank)]={'pooled':float(rows[prefix+f'_r{rank}_captured'].sum()/rows[prefix+'_energy'].sum()),'site_mean':float(rows[prefix+f'_r{rank}_site_ratio'].sum()/rows[prefix+'_count'].sum()),'head_macro':float(np.mean([heads[prefix+f'_h{h}'][str(rank)] for h in range(9)]))}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows)
    ix=rng.integers(cfg['rows'],size=(2000,cfg['rows']));contrasts={}
    def get(n,b,f,g,m='pooled'):return results[f'{n}_{b}_{f}_{g}']['16'][m]
    def boot(n,b,f,g,m):
        k=f'{n}_{b}_{f}_{g}';num=rows[k+'_r16_'+('captured' if m=='pooled' else 'site_ratio')];den=rows[k+('_energy' if m=='pooled' else '_count')];return num[ix].sum(1)/den[ix].sum(1)
    for base in cfg['baselines']:
        for g in cfg['regimes']:
            for m in ['pooled','site_mean']:
                gap=get('native',base,'actual',g,m)-get('native',base,'angle',g,m);bootgap=boot('native',base,'actual',g,m)-boot('native',base,'angle',g,m)
                v={'gap':gap,'paired_row_bootstrap95':np.quantile(bootgap,[.025,.975]).tolist(),'interactions':{}}
                for n in cases:
                    if n in ['native','joint']:continue
                    contrast=bootgap-boot(n,base,'actual',g,m)+boot(n,base,'angle',g,m)
                    v['interactions'][n]={'value':gap-get(n,base,'actual',g,m)+get(n,base,'angle',g,m),'paired_row_bootstrap95':np.quantile(contrast,[.025,.975]).tolist()}
                contrasts[f'{base}_{g}_{m}']=v
    retention=get('native','removed','actual','add')/get('native','intact','actual','remove')
    gates={'remaining_context_alignment':all(contrasts[f'removed_{g}_pooled']['gap']>=.08 and all(contrasts[f'removed_{g}_pooled']['interactions'][f'right{r}']['value']>=.05 for r in range(3)) for g in ['tangent','add']),
      'self_inclusion_collapse':contrasts['intact_tangent_pooled']['gap']>=.08 and abs(contrasts['removed_tangent_pooled']['gap'])<=.03,
      'same_secant_chart_retention':retention>=.8 and secant_error<1e-4}
    invariant=max(abs(results[k][str(r)][m]-results[k.replace('native_','joint_',1)][str(r)][m]) for k in results if k.startswith('native_') for r in cfg['ranks'] for m in ['pooled','site_mean','head_macro'])
    output={'config':cfg,'gates':gates,'results':results,'contrasts':contrasts,'head_results':heads,'draw_metrics':draw_metrics,'audits':audits,'head_replay':replay,'chart_orthogonality_error':orth_error,'secant_reversal_error':secant_error,'same_secant_retention':retention,'joint_metric_max_difference':invariant}
    STEM.with_suffix('.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    lines=['# Source self-inclusion versus remaining-context alignment','','| Baseline / edit | Actual | Shuffled | Gaussian | Angle matched | Actual equal-site | Angle equal-site |','|---|---:|---:|---:|---:|---:|---:|']
    for base in cfg['baselines']:
        for g in ['tangent','add','remove']:
            vals=[get('native',base,f,g) for f in families]+[get('native',base,f,g,'site_mean') for f in ['actual','angle']]
            lines.append('| '+base+' / '+g+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    lines+=['',f'Locked gates: {gates}. Same-secant retention: {retention:.3f}; reversal error: {secant_error:.3g}; joint metric difference: {invariant:.3g}.','',cfg['scope']]
    STEM.with_name(STEM.name.upper()+'.md').write_text('\n'.join(lines)+'\n');print(gates,flush=True)
if __name__=='__main__':main()
