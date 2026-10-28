"""H269 discovery-only nested core accounting; exact diagnostic, no readout fitting."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_response_input_bases import residual_pullback
from run_bilin18_l17_residual_term_accounting import residual_bilinear,mean_linear
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_core_width_accounting'
def outside_response(x,z,mean,q,L,R,D,p):
    a=x-mean;b=z-mean;a=a-(a@q)@q.T;b=b-(b@q)@q.T
    return residual_bilinear(a+b,b-a,L,R,D,p)
@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');old=load_program(ART/'bilin18_l17_residual_feature_objective_frozen.npz','cuda');mean=old['input_mean'].double()
    pre,_,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'rows':768});x=F.rms_norm(pre,(1152,));cov=torch.zeros(1152,1152,device='cuda',dtype=torch.float64);gen=torch.Generator(device='cuda').manual_seed(cfg['basis_seed'])
    for i in range(0,len(x),16):
        xx=x[i:i+16];v=torch.randn(len(xx),8,1152,generator=gen,device='cuda');g=residual_pullback(xx,v,L,R,D,p).reshape(-1,1152).double();cov+=g.T@g
    ev,q=torch.linalg.eigh(cov);bases={'jacobian':q.flip(1),'natural':old['basis'].double()};np.savez_compressed(STEM.with_name(STEM.name+'_bases.npz'),**{n:v[:,:128].cpu().numpy() for n,v in bases.items()},input_mean=mean.cpu().numpy(),jacobian_eigenvalues=ev.cpu().numpy());print('bases frozen',flush=True)
    L,R,D=[v.double() for v in [L,R,D]];p={k:v.double() for k,v in p.items()};pre,heads,replay=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,)).double();err=lambda z:B(z,z,L,R,D)-forward_normalized(z,p);e0=err(x);rows={};gen.manual_seed(cfg['seed']);max_replay=0.
    for h in cfg['heads']:
        actual=heads[:,h];source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;angle=angle_matched_payload(pre,actual,source,torch.randn(len(pre),128,generator=gen,device='cuda'))
        for label,pay in [('actual',actual),('angle',angle)]:
            for scale in cfg['scales']:
                z=F.rms_norm(pre-scale*pay,(1152,)).double();target=err(z)-e0;prefix=f'{label}_s{scale}_h{h}';energy=target.square().sum(-1);rows[prefix+'_energy']=energy.reshape(128,4).sum(1).cpu().numpy();fullquad=residual_bilinear(x+z-2*mean,z-x,L,R,D,p);lin=mean_linear(z-x,mean,L,R,D,p);max_replay=max(max_replay,float((lin+fullquad-target).norm()/target.norm()))
                for name,q in bases.items():
                    for k in cfg['widths']:
                        outside=outside_response(x,z,mean,q[:,:k],L,R,D,p);inside=target-outside;key=f'{prefix}_{name}{k}'
                        for metric,val in [('error',outside.square().sum(-1)),('alignment',(outside*target).sum(-1)),('kept_energy',inside.square().sum(-1)),('cross',(inside*outside).sum(-1))]:rows[key+'_'+metric]=val.reshape(128,4).sum(1).cpu().numpy()
        print('head',h,'complete',flush=True)
    ix=np.random.default_rng(cfg['seed']+1).integers(128,size=(2000,128));results={}
    for label in cfg['payloads']:
        for scale in cfg['scales']:
            prefix=f'{label}_s{scale}';total=lambda suffix:sum(rows[f'{prefix}_h{h}_{suffix}'] for h in cfg['heads']);e=total('energy');results[prefix]={}
            for name in bases:
                results[prefix][name]={}
                for k in cfg['widths']:
                    v=total(f'{name}{k}_error');results[prefix][name][k]={'fidelity':float(1-v.sum()/e.sum()),'document_bootstrap95':np.quantile(1-v[ix].sum(1)/e[ix].sum(1),[.025,.975]).tolist(),**{m:float(total(f'{name}{k}_{m}').sum()/e.sum()) for m in ['alignment','kept_energy','cross']}}
    gates={'compact32':any(all(results[f'actual_s{s}'][n][32]['fidelity']>=.8 for s in cfg['scales']) for n in bases),'both_inadequate64':all(results[f'actual_s{s}'][n][64]['fidelity']<.5 for s in cfg['scales'] for n in bases)}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'results':results,'gates':gates,'max_relative_replay':max_replay,'head_replay':replay},indent=2));print('GATES',gates,'replay',max_replay,flush=True)
if __name__=='__main__':main()
