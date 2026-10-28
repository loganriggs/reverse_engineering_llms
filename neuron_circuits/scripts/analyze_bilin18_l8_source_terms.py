#!/usr/bin/env python3
"""Exact, non-orthogonal source-addition decomposition; not semantic attribution."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_context_role_compression import B
from run_bilin18_l8_channel_coordination import canonical_weights,permuted_weights
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l8_source_terms'

def addition_terms(context,payload,L,R,D,epsilon=None):
    eps=torch.finfo(context.dtype).eps if epsilon is None else epsilon
    initial=context.square().mean(-1,keepdim=True)+eps
    final=(context+payload).square().mean(-1,keepdim=True)+eps
    return {'cross':2*B(context,payload,L,R,D)/final,
            'self':B(payload,payload,L,R,D)/final,
            'normalization':B(context,context,L,R,D)*(1/final-1/initial)}

@torch.inference_mode()
def main():
    torch.set_num_threads(6);data=np.load(ART/'bilin18_l8_source_self_inclusion_payloads.npz');pre=torch.from_numpy(data['pre']).cuda();hw=torch.from_numpy(data['heads']).cuda()
    model=load_full_model('cuda');model.blocks.float();block=model.blocks[8];weights=canonical_weights(*(m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]))
    frozen=np.load(ART/'bilin18_l8_canonical_channel_coordination_frozen.npz');cases={'native':('native',0),'joint':('joint',0),**{f'{k}{r}':(k,r) for k in ['right','down','direction'] for r in range(3)}}
    results={};rows={};maxerr=0.;double_replay=0.
    for name,(kind,rep) in cases.items():
        L,R,D=permuted_weights(*weights,torch.from_numpy(frozen[f'permutation{rep}']).cuda(),kind)
        for h in range(9):
            c=pre-hw[:,h];p=hw[:,h];terms=addition_terms(c,p,L,R,D);x=F.rms_norm(c,(1152,));y=F.rms_norm(pre,(1152,));target=B(y,y,L,R,D)-B(x,x,L,R,D)
            maxerr=max(maxerr,float((sum(terms.values())-target).norm()/target.norm()));energy=target.square().sum(-1)
            if name=='native':
                cd=c[:4].double();pd=p[:4].double();ld,rd,dd=[w.double() for w in [L,R,D]];eps=torch.finfo(torch.float32).eps
                terms64=addition_terms(cd,pd,ld,rd,dd,eps)
                xn=cd/(cd.square().mean(-1,keepdim=True)+eps).sqrt();yn=(cd+pd)/((cd+pd).square().mean(-1,keepdim=True)+eps).sqrt()
                truth=B(yn,yn,ld,rd,dd)-B(xn,xn,ld,rd,dd)
                double_replay=max(double_replay,float((sum(terms64.values())-truth).norm()/truth.norm()))
            metrics={'target_energy':float(energy.sum()),'terms':{},'cross_self_cosine':float((terms['cross']*terms['self']).sum()/(terms['cross'].norm()*terms['self'].norm()))}
            for label,t in terms.items():
                metrics['terms'][label]={'signed_projection':float((t*target).sum()/energy.sum()),'energy_ratio':float(t.square().sum()/energy.sum())}
                for suffix,v in [('dot',(t*target).sum(-1)),('energy',t.square().sum(-1))]:
                    key=f'{name}_{label}_{suffix}';rows.setdefault(key,np.zeros(32));rows[key]+=v.reshape(32,4).sum(1).cpu().numpy()
            key=f'{name}_target_energy';rows.setdefault(key,np.zeros(32));rows[key]+=energy.reshape(32,4).sum(1).cpu().numpy();results[f'{name}_h{h}']=metrics
    pooled={};rng=np.random.default_rng(20261109);ix=rng.integers(32,size=(2000,32))
    for name in cases:
        den=rows[f'{name}_target_energy'];pooled[name]={}
        for label in ['cross','self','normalization']:
            dot=rows[f'{name}_{label}_dot'];energy=rows[f'{name}_{label}_energy']
            pooled[name][label]={'signed_projection':float(dot.sum()/den.sum()),'energy_ratio':float(energy.sum()/den.sum()),'signed_projection_row_bootstrap95':np.quantile(dot[ix].sum(1)/den[ix].sum(1),[.025,.975]).tolist()}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows)
    output={'scope':'Descriptive exact algebra audit on H241 actual-addition pairs. No fresh replication. Decomposition convention uses final RMS denominator for cross/self terms and initial context for normalization term. Terms are nonorthogonal; energy ratios do not sum to one. Signed projections sum to one but are not unique causal credits.','formula':'F(c+h)-F(c)=2B(c,h)/s(c+h)+B(h,h)/s(c+h)+B(c,c)*(1/s(c+h)-1/s(c)); s(z)=mean(z^2)+eps.','float64_native_four_sites_per_head_replay_error':double_replay,'replay_relative_error_max':maxerr,'pooled':pooled,'head_results':results}
    STEM.with_suffix('.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    lines=['# Exact source response terms','',output['scope'],'',output['formula'],'','| Function | Cross signed projection | Self signed projection | Normalization signed projection | Cross energy / target | Self energy / target | Normalization energy / target |','|---|---:|---:|---:|---:|---:|---:|']
    for n,v in pooled.items():lines.append('| '+n+' | '+' | '.join(f'{v[t][m]:.3f}' for m in ['signed_projection','energy_ratio'] for t in ['cross','self','normalization'])+' |')
    lines+=['',f'Float32 maximum relative replay error {maxerr:.3g}; float64 native audit (four sites/head, float32 epsilon) {double_replay:.3g}.'];STEM.with_name(STEM.name.upper()+'.md').write_text('\n'.join(lines)+'\n');print('Replay',maxerr)
if __name__=='__main__':main()
