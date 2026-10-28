#!/usr/bin/env python3
"""Weight-only output tensor spectrum versus discovery response geometry."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_source_support_factorial import collect_sources
from run_bilin18_l8_context_role_compression import B
from run_bilin18_l8_execution_alignment import angle_matched_payload
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition'

def output_tensor_gram(L,R,D):
    cross=L@R.T
    gram=.5*((L@L.T)*(R@R.T)+cross*cross.T)
    out=(D@gram)@D.T
    return (out+out.T)*.5

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs/bilin18_depth_source_geometry.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();output={};frozen={};rows={}
    for layer in cfg['layers']:
        block=model.blocks[layer];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];gram=output_tensor_gram(L,R,D);eig,u=torch.linalg.eigh(gram);eig=eig.flip(0);basis=u.flip(1).T[:64];frozen[f'L{layer}_basis']=basis.cpu().numpy()
        record={'raw_tensor_energy_capture':{str(r):float(eig[:r].sum()/eig.sum()) for r in cfg['ranks']},'minimum_eigenvalue_relative':float(eig[-1]/eig[0]),'responses':{}}
        pre,hw,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),{**cfg,'layer':layer});x=F.rms_norm(pre,(1152,));fx=B(x,x,L,R,D)
        for h in range(9):
            ref=hw[:,h];source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;gen=torch.Generator(device='cuda').manual_seed(cfg['seed']+100*layer+h)
            payloads={'actual':[ref],'angle':[angle_matched_payload(pre,ref,source,torch.randn(len(pre),128,generator=gen,device='cuda')) for _ in range(2)]}
            for fam,draws in payloads.items():
                prefix=f'L{layer}_{fam}'
                for p in draws:
                    y=F.rms_norm(pre-p,(1152,));target=B(y,y,L,R,D)-fx;energy=target.square().sum(-1);valid=energy>1e-20
                    for label,v in [('energy',energy),('count',valid.float())]:
                        key=prefix+'_'+label;rows.setdefault(key,np.zeros(cfg['rows']));rows[key]+=v.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
                    for r in cfg['ranks']:
                        cap=(target@basis[:r].T).square().sum(-1);ratio=torch.where(valid,cap/energy.clamp_min(1e-20),torch.zeros_like(energy))
                        for label,v in [('captured',cap),('site_ratio',ratio)]:
                            key=prefix+f'_r{r}_'+label;rows.setdefault(key,np.zeros(cfg['rows']));rows[key]+=v.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
        for fam in ['actual','angle']:
            prefix=f'L{layer}_{fam}';record['responses'][fam]={str(r):{'pooled':float(rows[prefix+f'_r{r}_captured'].sum()/rows[prefix+'_energy'].sum()),'site_mean':float(rows[prefix+f'_r{r}_site_ratio'].sum()/rows[prefix+'_count'].sum())} for r in cfg['ranks']}
        output[str(layer)]=record;print(layer,record,flush=True)
    np.savez_compressed(ART/'bilin18_depth_tensor_output_frozen.npz',**frozen);np.savez_compressed(ART/'bilin18_depth_tensor_output_rows.npz',**rows)
    (ART/'bilin18_depth_tensor_output.json').write_text(json.dumps({'scope':'Adaptive diagnostic after H244. Weight-only basis from exact symmetric bilinear output tensor Gram; no data fitting. Frobenius input metric is Euclidean, not natural-state covariance. Same H244 intact actual and angle-control payloads. Gauge-invariant to factor signed scaling, exchange and joint channel permutation; residual coordinate metric still matters.','results':output},indent=2))
if __name__=='__main__':main()
