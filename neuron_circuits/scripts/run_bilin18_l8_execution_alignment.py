#!/usr/bin/env python3
"""Within-state payload controls isolate execution alignment from operator geometry."""
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

ROOT=Path(__file__).resolve().parents[1];STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_execution_alignment'

def shifted_indices(rows,positions,shift,device):
    return (((torch.arange(rows,device=device)+shift)%rows)[:,None]*positions+torch.arange(positions,device=device)[None]).flatten()

def angle_matched_payload(pre,ref,source,noise):
    axis=pre@source
    axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-30)
    coeff=ref@source
    parallel=(coeff*axis).sum(-1,keepdim=True)*axis
    perpendicular=noise-(noise*axis).sum(-1,keepdim=True)*axis
    perpendicular=perpendicular/perpendicular.norm(dim=-1,keepdim=True).clamp_min(1e-30)
    return (parallel+(coeff-parallel).norm(dim=-1,keepdim=True)*perpendicular)@source.T

@torch.inference_mode()
def main():
    global STEM
    if '--angle-control' in sys.argv:STEM=STEM.with_name('bilin18_l8_angle_matched_alignment')
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());torch.manual_seed(cfg['seed'])
    model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[8]
    weights=canonical_weights(*(m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]))
    frozen=np.load(STEM.parent/'bilin18_l8_canonical_channel_coordination_frozen.npz')
    permutations={r:torch.from_numpy(frozen[f'permutation{r}']).cuda() for r in range(3)}
    cases={'native':('native',0),'joint':('joint',0),**{f'{kind}{r}':(kind,r) for kind in ['right','down','direction'] for r in range(3)}}
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');pre,hw,replay=collect_sources(model,corpus,cfg);x=F.rms_norm(pre,(1152,));N=len(x)
    sources=[torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
    payloads={};payload_audit={};arrays={};rng=np.random.default_rng(cfg['seed']);norm_error=0.;span_error=0.
    for h in range(9):
        source=sources[h];ref=hw[:,h];payloads[h]={'actual':[ref],'shuffled':[],'gaussian':[]}
        if cfg.get('angle_control'):payloads[h]['angle_gaussian']=[]
        shifts=rng.choice(np.arange(1,cfg['rows']),size=cfg['sham_draws'],replace=False)
        payload_audit[str(h)]={'row_shifts':shifts.tolist()}
        for draw,shift in enumerate(shifts):
            idx=shifted_indices(cfg['rows'],len(cfg['positions']),int(shift),'cuda');payloads[h]['shuffled'].append(norm_match(ref[idx],ref))
            gen=torch.Generator(device='cuda').manual_seed(cfg['seed']+100*h+draw)
            payloads[h]['gaussian'].append(norm_match(torch.randn(N,128,generator=gen,device='cuda')@source.T,ref))
            if cfg.get('angle_control'):
                noise=torch.randn(N,128,generator=gen,device='cuda')
                payloads[h]['angle_gaussian'].append(angle_matched_payload(pre,ref,source,noise))
        for family,draws in payloads[h].items():
            for i,p in enumerate(draws):
                norm_error=max(norm_error,float((p.norm(dim=-1)-ref.norm(dim=-1)).abs().max()/ref.norm(dim=-1).max()))
                span_error=max(span_error,float((p-(p@source)@source.T).norm()/p.norm()))
                arrays[f'h{h}_{family}{i}']=p.cpu().numpy()
    families=list(payloads[0])
    angle_audit={}
    if cfg.get('angle_control'):
        for h in range(9):
            ref=hw[:,h]
            for i,p in enumerate(payloads[h]['angle_gaussian']):
                angle_audit[f'h{h}_draw{i}']={'dot_error':float(((pre*p).sum(-1)-(pre*ref).sum(-1)).abs().div(pre.norm(dim=-1)*ref.norm(dim=-1)).max()),'rms_denominator_error':max(float(((pre-a*p).norm(dim=-1)-(pre-a*ref).norm(dim=-1)).abs().div((pre-a*ref).norm(dim=-1)).max()) for a in [.05,1.]),'perpendicular_energy_fraction_mean':float((1-((ref@sources[h])*(pre@sources[h])).sum(-1).square()/((ref@sources[h]).square().sum(-1)*(pre@sources[h]).square().sum(-1))).mean())}
    np.savez_compressed(STEM.with_name(STEM.name+'_payloads.npz'),pre=pre.cpu().numpy(),**arrays)
    print('All payload draws fixed before response evaluation',flush=True)
    rows={};heads={};draw_metrics={};orth_error=0.
    for name,(kind,rep) in cases.items():
        L,R,D=permuted_weights(*weights,permutations[rep],kind);fx=B(x,x,L,R,D)
        for h in range(9):
            source=sources[h];Js=[];charts=[]
            for offset in range(0,N,cfg['matrix_batch']):
                J=source_jacobian(pre[offset:offset+cfg['matrix_batch']],L,R,D,source);u,_,_=torch.linalg.svd(J,full_matrices=False);u=torch.linalg.qr(u[:,:,:64]).Q
                Js.append(J);charts.append(u.transpose(1,2));orth_error=max(orth_error,float((u.transpose(1,2)@u-torch.eye(64,device='cuda')).abs().max()))
            J=torch.cat(Js);chart=torch.cat(charts)
            for family,draws in payloads[h].items():
                head_accum={}
                for draw,p in enumerate(draws):
                    tangent=-torch.einsum('ndr,nr->nd',J,p@source)
                    for regime in cfg['regimes']:
                        if regime=='tangent':target=tangent
                        else:
                            scale=.05 if regime=='small' else 1.
                            y=F.rms_norm(pre-scale*p,(1152,));target=B(y,y,L,R,D)-fx
                        energy=target.square().sum(-1);valid=energy>1e-20
                        prefix=f'{name}_{family}_{regime}';head_accum.setdefault(regime,{'energy':0.,'captured':{r:0. for r in cfg['ranks']}});head_accum[regime]['energy']+=float(energy.sum())
                        for label,values in [('energy',energy),('count',valid.float())]:
                            key=prefix+'_'+label;rows.setdefault(key,np.zeros(cfg['rows']));rows[key]+=values.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
                        for rank in cfg['ranks']:
                            captured=project(target,chart[:,:rank]).square().sum(-1);ratio=torch.where(valid,captured/energy.clamp_min(1e-20),torch.zeros_like(energy))
                            head_accum[regime]['captured'][rank]+=float(captured.sum())
                            for label,values in [('captured',captured),('site_ratio',ratio)]:
                                key=prefix+f'_r{rank}_'+label;rows.setdefault(key,np.zeros(cfg['rows']));rows[key]+=values.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
                            draw_metrics[f'{name}_h{h}_{family}{draw}_{regime}_r{rank}']={'pooled':float(captured.sum()/energy.sum()),'site_mean':float(ratio.sum()/valid.sum())}
                for regime,acc in head_accum.items():
                    heads[f'{name}_h{h}_{family}_{regime}']={str(r):v/acc['energy'] for r,v in acc['captured'].items()}
            print(name,'head',h,'complete',flush=True)
        # Save progress after each completed local function, without claiming completion.
        STEM.with_name(STEM.name+'_partial.json').write_text(json.dumps({'completed':name,'head_results':heads},indent=2))
    results={}
    for name in cases:
        for family in families:
            for regime in cfg['regimes']:
                prefix=f'{name}_{family}_{regime}';results[prefix]={}
                for rank in cfg['ranks']:
                    results[prefix][str(rank)]={'pooled':float(rows[prefix+f'_r{rank}_captured'].sum()/rows[prefix+'_energy'].sum()),
                        'site_mean':float(rows[prefix+f'_r{rank}_site_ratio'].sum()/rows[prefix+'_count'].sum()),
                        'head_macro':float(np.mean([heads[f'{name}_h{h}_{family}_{regime}'][str(rank)] for h in range(9)])),
                        'valid_sites':int(rows[prefix+'_count'].sum())}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows)
    ix=rng.integers(cfg['rows'],size=(2000,cfg['rows']));contrasts={}
    def boot(name,family,regime,aggregation):
        prefix=f'{name}_{family}_{regime}'
        numerator=rows[prefix+'_r16_'+('captured' if aggregation=='pooled' else 'site_ratio')];denominator=rows[prefix+('_energy' if aggregation=='pooled' else '_count')]
        return numerator[ix].sum(1)/denominator[ix].sum(1)
    for name in cases:
        if name in ['native','joint']:continue
        contrasts[name]={}
        for regime in ['tangent','full']:
            contrasts[name][regime]={}
            for aggregation in ['pooled','site_mean']:
                difference=boot('native','actual',regime,aggregation)-boot(name,'actual',regime,aggregation)-boot('native','gaussian',regime,aggregation)+boot(name,'gaussian',regime,aggregation)
                get=lambda n,f:results[f'{n}_{f}_{regime}']['16'][aggregation]
                contrasts[name][regime][aggregation]={'actual_specific_native_advantage':get('native','actual')-get(name,'actual')-get('native','gaussian')+get(name,'gaussian'),'paired_row_bootstrap95':np.quantile(difference,[.025,.975]).tolist()}
    get=lambda n,f,g:results[f'{n}_{f}_{g}']['16']['pooled']
    gates={'execution_alignment':all(contrasts[f'right{r}'][g]['pooled']['actual_specific_native_advantage']>=.08 for r in range(3) for g in ['tangent','full']),
        'context_pairing':all(get('native','actual',g)-get('native','shuffled',g)>=.08 for g in ['tangent','full']),
        'marginal_anisotropy':all(get('native','shuffled',g)-get('native','gaussian',g)>=.05 for g in ['tangent','full']),
        'intrinsic_operator':all(get('native','gaussian',g)-get(f'right{r}','gaussian',g)>=.10 for r in range(3) for g in ['tangent','full'])}
    joint_difference=max(abs(results['native_'+f+'_'+g][str(r)][m]-results['joint_'+f+'_'+g][str(r)][m]) for f in families for g in cfg['regimes'] for r in cfg['ranks'] for m in ['pooled','site_mean','head_macro'])
    angle_contrasts={}
    if cfg.get('angle_control'):
        for regime in ['tangent','full']:
            angle_contrasts[regime]={}
            for aggregation in ['pooled','site_mean']:
                gap=boot('native','actual',regime,aggregation)-boot('native','angle_gaussian',regime,aggregation)
                entry={'native_gap':results[f'native_actual_{regime}']['16'][aggregation]-results[f'native_angle_gaussian_{regime}']['16'][aggregation],'paired_row_bootstrap95':np.quantile(gap,[.025,.975]).tolist(),'interactions':{}}
                for k in range(3):
                    n=f'right{k}';interaction=gap-boot(n,'actual',regime,aggregation)+boot(n,'angle_gaussian',regime,aggregation)
                    entry['interactions'][n]={'value':entry['native_gap']-results[f'{n}_actual_{regime}']['16'][aggregation]+results[f'{n}_angle_gaussian_{regime}']['16'][aggregation],'paired_row_bootstrap95':np.quantile(interaction,[.025,.975]).tolist()}
                angle_contrasts[regime][aggregation]=entry
        gates['simple_angle_geometry']=all(abs(angle_contrasts[g]['pooled']['native_gap'])<=.03 and get('native','actual',g)-get('native','gaussian',g)>=.08 for g in ['tangent','full'])
        gates['residual_pairing']=all(angle_contrasts[g]['pooled']['native_gap']>=.08 and all(v['value']>=.05 for v in angle_contrasts[g]['pooled']['interactions'].values()) for g in ['tangent','full'])
    output={'angle_audit':angle_audit,'angle_contrasts':angle_contrasts,'config':cfg,'gates':gates,'head_replay':replay,'payload_norm_relative_error':norm_error,'payload_span_relative_error':span_error,'chart_orthogonality_error':orth_error,'joint_metric_max_difference':joint_difference,'payload_audit':payload_audit,'results':results,'head_results':heads,'draw_metrics':draw_metrics,'contrasts':contrasts}
    STEM.with_suffix('.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    lines=['# Actual source execution versus intrinsic local concentration','','Same states, source spans and per-site raw norms across payload families. No fitted decoder; all outputs are evaluated.','','| Local function | Actual16 | Shuffled16 | Gaussian16 | Actual site-mean16 | Shuffled site-mean16 | Gaussian site-mean16 |','|---|---:|---:|---:|---:|---:|---:|']
    if cfg.get('angle_control'):lines=['# Angle-matched source execution controls','','| Local function | Actual | Shuffled | Gaussian | Angle Gaussian | Actual site | Shuffled site | Gaussian site | Angle Gaussian site |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in cases:
        values=[results[f'{name}_{f}_full']['16'][m] for m in ['pooled','site_mean'] for f in families]
        lines.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines+=['',f'Locked gates: {gates}. Joint invariant metric change: {joint_difference:.3g}.','',cfg['scope']]
    STEM.with_name(STEM.name.upper()+'.md').write_text('\n'.join(lines)+'\n');print(gates,flush=True)

if __name__=='__main__':main()
