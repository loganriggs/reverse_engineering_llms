"""H270 exact basis x interaction-family factorial, frozen H269 bases."""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_residual_term_accounting import residual_bilinear,mean_linear
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_core_interaction_factorial'
def quadratic_terms(x,z,mean,q,L,R,D,p):
    a=x-mean;b=z-mean;ca=(a@q)@q.T;cb=(b@q)@q.T;ra=a-ca;rb=b-cb
    bil=lambda a,b:residual_bilinear(a,b,L,R,D,p)
    return torch.stack([bil(ca+cb,cb-ca),bil(ca+cb,rb-ra)+bil(cb-ca,ra+rb),bil(ra+rb,rb-ra)],1)
@torch.inference_mode()
def main():
    global STEM
    if len(sys.argv)>1:STEM=ART/sys.argv[1]
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[17];L,R,D=[m.weight.double() for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda',torch.float64);f=load_program(ART/'bilin18_l17_core_width_accounting_bases.npz','cuda',torch.float64);linear=load_program(ART/'bilin18_l17_residual_tensor_partners_frozen.npz','cuda',torch.float64)['linear_basis'];mean=f['input_mean'];pre,heads,replay=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,)).double();err=lambda z:B(z,z,L,R,D)-forward_normalized(z,p);e0=err(x);rows={};gen=torch.Generator(device='cuda').manual_seed(cfg['seed']);max_replay=0.
    for h in cfg['heads']:
        actual=heads[:,h];source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;angle=angle_matched_payload(pre,actual,source,torch.randn(len(pre),128,generator=gen,device='cuda'))
        for label,pay in [('actual',actual),('angle',angle)]:
            for scale in cfg['scales']:
                z=F.rms_norm(pre-scale*pay,(1152,)).double();target=err(z)-e0;prefix=f'{label}_s{scale}_h{h}';rows[prefix+'_energy']=target.square().sum(-1).reshape(128,4).sum(1).cpu().numpy();dx=z-x;lins={'full':mean_linear(dx,mean,L,R,D,p),'624':mean_linear((dx@linear)@linear.T,mean,L,R,D,p)}
                for name in cfg['families']:
                    for k in cfg['widths']:
                        t=quadratic_terms(x,z,mean,f[name][:,:k],L,R,D,p);max_replay=max(max_replay,float((lins['full']+t.sum(1)-target).norm()/target.norm()));key=f'{prefix}_{name}{k}'
                        rows[key+'_gram']=torch.einsum('ntd,nud->ntu',t,t).reshape(128,4,3,3).sum(1).cpu().numpy()
                        for lin,lv in lins.items():
                            for variant,quad in [('within',t[:,0]),('all',t[:,:2].sum(1))]:rows[f'{key}_{lin}_{variant}']=(lv+quad-target).square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
        print('head',h,'complete',flush=True)
    ix=np.random.default_rng(cfg['seed']+1).integers(128,size=(2000,128));results={};contrasts={}
    for label in cfg['payloads']:
        for scale in cfg['scales']:
            prefix=f'{label}_s{scale}';total=lambda suffix:sum(rows[f'{prefix}_h{h}_{suffix}'] for h in cfg['heads']);e=total('energy');results[prefix]={};contrasts[prefix]={}
            stat=lambda d:{'gain':float(d.sum()/e.sum()),'paired_document_bootstrap95':np.quantile(d[ix].sum(1)/e[ix].sum(1),[.025,.975]).tolist()}
            for k in cfg['widths']:
                for name in cfg['families']:
                    for lin in ['full','624']:
                        for variant in ['within','all']:
                            key=f'{name}{k}_{lin}_{variant}';er=total(key);results[prefix][key]={'fidelity':float(1-er.sum()/e.sum()),'document_bootstrap95':np.quantile(1-er[ix].sum(1)/e[ix].sum(1),[.025,.975]).tolist()}
                    results[prefix][f'{name}{k}_quadratic_gram']=(total(f'{name}{k}_gram').sum(0)/e.sum()).tolist()
                for lin in ['full','624']:
                    d={v:total(f'jacobian{k}_{lin}_{v}')-total(f'natural{k}_{lin}_{v}') for v in ['within','all']}
                    for v in d:contrasts[prefix][f'{k}_{lin}_{v}']=stat(d[v])
                    contrasts[prefix][f'{k}_{lin}_interaction']=stat(d['all']-d['within'])
    gates={}
    for lin in ['full','624']:
        gates[f'exact_reversal_{lin}']=all(contrasts[f'actual_s{s}'][f'32_{lin}_within']['paired_document_bootstrap95'][1]<0 and contrasts[f'actual_s{s}'][f'32_{lin}_all']['paired_document_bootstrap95'][0]>0 for s in cfg['scales'])
        gates[f'interaction_gain_{lin}']=all(contrasts[f'actual_s{s}'][f'32_{lin}_interaction']['gain']>=.05 and contrasts[f'actual_s{s}'][f'32_{lin}_interaction']['paired_document_bootstrap95'][0]>0 for s in cfg['scales'])
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'results':results,'contrasts':contrasts,'gates':gates,'max_relative_replay':max_replay,'head_replay':replay},indent=2));print('GATES',gates,'replay',max_replay,flush=True)
if __name__=='__main__':main()
