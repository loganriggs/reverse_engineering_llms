"""H295 proxy-weighted partner ranks for the frozen96-dimensional read span."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program
from run_bilin18_l17_residual_tensor_partners import residual_bilinear_matrix
from run_bilin18_l17_partner_compression import cross_response,core_metric
from run_bilin18_l17_core_interaction_factorial import quadratic_terms
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_conditional_balanced_eval'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[17];L,R,D=[m.weight.double() for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda',torch.float64);f=load_program(ART/'bilin18_l17_conditional_balanced_fit_frozen.npz','cuda',torch.float64);core=torch.linalg.qr(f['core']).Q;mu=f['mean'];partners={n:torch.linalg.qr(f[n]-core@(core.T@f[n])).Q for n in ['weighted','random0','optimized']};old=load_program(ART/'bilin18_l17_conditional_direct_fit_frozen.npz','cuda',torch.float64);partners['pooled']=torch.linalg.qr(old['optimized']-core@(core.T@old['optimized'])).Q;audit={n:{'overlap':float((core.T@q).abs().max()),'orthogonality':float((q.T@q-torch.eye(32,device='cuda')).abs().max())} for n,q in partners.items()}
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,)).double();gen=torch.Generator(device='cuda').manual_seed(cfg['seed']);rows={};replay=0.
 def save(k,v):rows[k]=v.reshape(128,4).sum(1).cpu().numpy()
 for h in range(9):
  source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;actual=heads[:,h];angle=angle_matched_payload(pre,actual,source,torch.randn(len(pre),128,generator=gen,device='cuda'))
  for label,pay in [('actual',actual),('angle',angle)]:
   z=F.rms_norm(pre-pay,(1152,)).double();cross=quadratic_terms(x,z,mu,core,L,R,D,p)[:,1];prefix=f'{label}_h{h}';save(prefix+'_energy',cross.square().sum(-1))
   for name,q in partners.items():
    for k in cfg['ranks']:
     pred=cross_response(x,z,mu,core,q[:,:k],L,R,D,p);save(f'{prefix}_{name}{k}_error',(pred-cross).square().sum(-1))
  print('head',h,'complete',flush=True)
 ix=np.random.default_rng(cfg['seed']+1).integers(128,size=(2000,128));results={}
 for label in ['actual','angle']:
  energy=sum(rows[f'{label}_h{h}_energy'] for h in range(9));res={}
  for name in partners:
   for k in cfg['ranks']:
    err=sum(rows[f'{label}_h{h}_{name}{k}_error'] for h in range(9));res[f'{name}{k}']={'cross_fidelity':float(1-err.sum()/energy.sum()),'paired95':np.quantile(1-err[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
  results[label]=res
 contrasts={}
 for label in ['actual','angle']:
  energy=sum(rows[f'{label}_h{h}_energy'] for h in range(9));gain=sum(rows[f'{label}_h{h}_weighted32_error']-rows[f'{label}_h{h}_optimized32_error'] for h in range(9));contrasts[label]={'gain':float(gain.sum()/energy.sum()),'paired95':np.quantile(gain[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
 pooled_contrasts={}
 for label in ['actual','angle']:
  energy=sum(rows[f'{label}_h{h}_energy'] for h in range(9));gain=sum(rows[f'{label}_h{h}_pooled32_error']-rows[f'{label}_h{h}_optimized32_error'] for h in range(9));pooled_contrasts[label]={'gain':float(gain.sum()/energy.sum()),'paired95':np.quantile(gain[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
 gates={'balanced_gain':all(c['gain']>=.05 and c['paired95'][0]>0 for c in contrasts.values()),'compact80':all(v['optimized32']['cross_fidelity']>=.8 for v in results.values()),'retention':all(c['paired95'][0]>=-.01 for c in contrasts.values()),'pooled_tradeoff_repair':pooled_contrasts['actual']['gain']>=.02 and pooled_contrasts['actual']['paired95'][0]>0 and pooled_contrasts['angle']['paired95'][0]>=-.01}
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'results':results,'gates':gates,'basis_audits':audit,'contrasts':contrasts,'pooled_contrasts':pooled_contrasts,'bilinear_output_coefficients':{str(k):96*k*1152 for k in cfg['ranks']}},indent=2)+'\n');print('GATES',gates,'RESULTS',results,flush=True)
if __name__=='__main__':main()
