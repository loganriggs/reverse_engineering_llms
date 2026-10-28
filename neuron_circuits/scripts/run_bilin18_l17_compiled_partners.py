#!/usr/bin/env python3
"""Compile output-projected cross-context quadratic maps into fixed read programs."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_source_support_factorial import collect_sources
from run_bilin18_l8_context_role_compression import B,state_rules
from run_bilin18_l8_execution_alignment import angle_matched_payload
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_compiled_partners'

def compile_partners(L,R,D,context,mu,outputs,tolerance=1e-10):
    # Compile in double precision; normalized columns prevent arbitrary unit gains
    # from causing useful context or mean-linear reads to be dropped by rank tests.
    L,R,D,context,mu,outputs=[t.double() for t in [L,R,D,context,mu,outputs]]
    qp=[];qm=[]
    for u in outputs:
        a=u@D
        qp.append(.5*(L.T@(a[:,None]*(R@context))+R.T@(a[:,None]*(L@context))))
        qm.append(.5*(L.T@(a*(R@mu))+R.T@(a*(L@mu))))
    columns=torch.cat([context]+qp+[torch.stack(qm,1)],1);norms=columns.norm(dim=0);active=norms>0;scaled=columns[:,active]/norms[active]
    U,s,_=torch.linalg.svd(scaled,full_matrices=False);rank=int((s>s[0]*tolerance).sum());U=U[:,:rank];a=U.T@context;forms=[];linear=[]
    for q,v in zip(qp,qm):
        c=U.T@q;forms.append(a@c.T+c@a.T-a@(context.T@q)@a.T);linear.append(2*(U.T@v))
    return {'reads':U,'offset':mu@U,'forms':torch.stack(forms),'linear':torch.stack(linear),'outputs':outputs,'bias':torch.zeros(D.shape[0],device=D.device,dtype=torch.double)},s

def coefficients(x,program):
    z=x@program['reads']-program['offset']
    return torch.einsum('ni,oij,nj->no',z,program['forms'],z)+z@program['linear'].T

def normalized_forward(x,program):return coefficients(x,program)@program['outputs']+program['bias']

def raw_forward(raw,program):
    # Match the float32 native experiment's epsilon even when caller uses float64.
    x=raw/(raw.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
    return normalized_forward(x,program)

@torch.inference_mode()
def main():
    torch.set_num_threads(6);spec=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());parent=json.loads((ART/'bilin18_depth_linear_context.json').read_text());cfg=parent['config'];layer=spec['layer'];model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[layer];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]]
    frozen=np.load(ART/'bilin18_depth_linear_context_frozen.npz');context=torch.from_numpy(frozen[f'L{layer}_pca'][:,:spec['context_rank']]).cuda();mu=torch.from_numpy(frozen[f'L{layer}_mean']).cuda();outputfile=np.load(ART/'bilin18_depth_source_geometry_frozen.npz');outputs=torch.from_numpy(outputfile[f'L{layer}_global']).cuda()
    dc,_,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'layer':layer});dx=F.rms_norm(dc,(1152,));discovery_write=B(dx,dx,L,R,D);mean_write=discovery_write.mean(0)
    pre,hw,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),{**cfg,'layer':layer});x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);native_variance=(native-mean_write).square().sum();programs={};meta={};results={};rows={};native_count=sum(t.numel() for t in [L,R,D])+1152
    for m in spec['output_ranks']:
        prog64,s=compile_partners(L,R,D,context,mu,outputs[:m],spec['span_relative_tolerance']);prog64['bias']=mean_write.double()-coefficients(dx.double(),prog64).mean(0)@prog64['outputs']
        direct=state_rules(x.double(),mu.double(),context.T.double(),context.double(),L.double(),R.double(),D.double())['cross']@outputs[:m].double().T
        refcoeff=coefficients(x.double(),prog64);error64=float((refcoeff-direct).norm()/direct.norm());prog={k:v.float() for k,v in prog64.items()};error32=float((coefficients(x,prog).double()-direct).norm()/direct.norm());programs[m]=prog
        path=STEM.with_name(f'{STEM.name}_m{m}.npz');np.savez_compressed(path,**{k:v.cpu().numpy() for k,v in prog.items()})
        # Verify inference from serialized coefficients, with no original MLP tensors.
        disk=np.load(path);loaded={k:torch.from_numpy(disk[k]).cuda() for k in disk.files};serialization=float((normalized_forward(x,prog)-normalized_forward(x,loaded)).abs().max())
        count=sum(v.numel() for v in prog.values());prediction=normalized_forward(x,prog);rank=prog['reads'].shape[1]
        meta[str(m)]={'reads':rank,'read_bound':spec['context_rank']+m*spec['context_rank']+m,'stored_scalars':count,'native_scalars':native_count,'storage_fraction':count/native_count,'float32_bytes':4*count,'equivalence64_relative_rms':error64,'equivalence32_relative_rms':error32,'serialized_max_difference':serialization,'natural_write_r2':float(1-(prediction-native).square().sum()/native_variance),'span_smallest_kept_relative_singular':float(s[rank-1]/s[0]),'dense_mac_estimate':1152*rank+m*rank*rank+m*rank+m*1152,'native_dense_mac_estimate':3*1152*4608}
        print('compiled',m,meta[str(m)],flush=True)
    # Freeze all candidates before response scoring.
    for h in range(9):
        ref=hw[:,h];source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;gen=torch.Generator(device='cuda').manual_seed(cfg['seed']+100*layer+h);noises=[torch.randn(len(pre),128,generator=gen,device='cuda') for _ in range(cfg['sham_draws'])]
        for base in cfg['baselines']:
            raw=pre if base=='intact' else pre-ref;sign=-1 if base=='intact' else 1;xn=F.rms_norm(raw,(1152,));fx=B(xn,xn,L,R,D);payloads={'actual':[ref],'angle':[angle_matched_payload(raw,ref,source,n) for n in noises]}
            initial={m:normalized_forward(xn,prog) for m,prog in programs.items()}
            for fam,draws in payloads.items():
                for scale in cfg['scales']:
                    prefix=f'{base}_{fam}_s{scale:g}'
                    for p in draws:
                        yn=F.rms_norm(raw+sign*scale*p,(1152,));target=B(yn,yn,L,R,D)-fx;energy=target.square().sum(-1);valid=energy>1e-20
                        for label,v in [('energy',energy),('count',valid.float())]:
                            key=prefix+'_'+label;rows.setdefault(key,np.zeros(cfg['rows']));rows[key]+=v.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
                        for m,prog in programs.items():
                            prediction=normalized_forward(yn,prog)-initial[m];error=(prediction-target).square().sum(-1);ratio=torch.where(valid,error/energy.clamp_min(1e-20),torch.zeros_like(energy))
                            for label,v in [('error',error),('site_ratio',ratio)]:
                                key=prefix+f'_m{m}_'+label;rows.setdefault(key,np.zeros(cfg['rows']));rows[key]+=v.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
        print('head',h,'complete',flush=True)
    rng=np.random.default_rng(20261116);ix=rng.integers(cfg['rows'],size=(2000,cfg['rows']))
    for key,e in rows.items():
        if not key.endswith('_error'):continue
        name=key[:-6];prefix='_'.join(name.split('_')[:3]);den=rows[prefix+'_energy'];results[name]={'pooled':float(1-e.sum()/den.sum()),'site_mean':float(1-rows[name+'_site_ratio'].sum()/rows[prefix+'_count'].sum()),'document_bootstrap95':np.quantile(1-e[ix].sum(1)/den[ix].sum(1),[.025,.975]).tolist()}
    def passes(m,fams):return meta[str(m)]['natural_write_r2']>=.9 and meta[str(m)]['reads']<=288 and meta[str(m)]['storage_fraction']<.1 and all(results[f'{b}_{f}_s{s:g}_m{m}']['pooled']>=.9 for b in cfg['baselines'] for f in fams for s in cfg['scales'])
    gates={'compiler_equivalence':all(v['equivalence32_relative_rms']<=1e-4 for v in meta.values()),'compact_actual_forward':any(passes(m,['actual']) for m in programs),'compact_universal_forward':any(passes(m,['actual','angle']) for m in programs)}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':spec,'parent_config':cfg,'gates':gates,'programs':meta,'results':results},indent=2))
    lines=['# Compiled fixed quadratic forward programs','','| Output directions | Normalized linear reads | Stored scalars | Native storage fraction | Natural-write R2 | Actual full-response fidelity | Angle full-response fidelity |','|---|---:|---:|---:|---:|---:|---:|']
    for m,v in meta.items():lines.append(f"| {m} | {v['reads']} | {v['stored_scalars']} | {v['storage_fraction']:.4f} | {v['natural_write_r2']:.3f} | {results[f'intact_actual_s1_m{m}']['pooled']:.3f} | {results[f'intact_angle_s1_m{m}']['pooled']:.3f} |")
    lines+=['',f'Locked gates: {gates}.','',spec['scope']];STEM.with_name(STEM.name.upper()+'.md').write_text('\n'.join(lines)+'\n');print(gates,flush=True)
if __name__=='__main__':main()
