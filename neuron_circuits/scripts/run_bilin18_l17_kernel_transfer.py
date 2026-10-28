"""H316 frozen18x18 interaction-readout transfer matrix."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program
from run_bilin18_l17_partner_compression import cross_response
from run_bilin18_l17_core_interaction_factorial import quadratic_terms
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_kernel_transfer'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((folder/n).read_bytes()).hexdigest()==digest
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));gen=torch.Generator(device='cuda').manual_seed(cfg['fresh_payload_seed']);zs=[]
 for h in range(9):
  source=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;pay=angle_matched_payload(pre,heads[:,h],source,torch.randn(len(pre),128,generator=gen,device='cuda'));zs.append(F.rms_norm(pre-pay,(1152,)))
 zs=[F.rms_norm(pre-heads[:,h],(1152,)) for h in range(9)]+zs
 def coords(v):
  w=v.double()-mu.double();u=w@c.double();return u,w-u@c.double().T-(w@q.double())@q.double().T
 u,v=coords(x);features=[];targets=[];rows={}
 for target,z in enumerate(zs):
  w,t=coords(z);features.append((w,t));y=torch.cat([quadratic_terms(x[i:i+256],z[i:i+256],mu,c,L,R,D,p)[:,1]-cross_response(x[i:i+256],z[i:i+256],mu,c,q,L,R,D,p) for i in range(0,512,256)]).double();targets.append(y);rows[f't{target}_energy']=y.square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
 for source in range(18):
  f=np.load(folder/f'g{source}.npz');wref=torch.tensor(f['uz'],device='cuda');tref=torch.tensor(f['vz'],device='cuda');coef=torch.tensor(f['interaction_coef'],device='cuda');scale=float(f['interaction_kernel_scale'])
  for target,(w,t) in enumerate(features):
   K=((w@wref.T)*(t@tref.T)+(u@ux.T)*(v@vx.T)-(w@ux.T)*(t@vx.T)-(u@wref.T)*(v@tref.T))/scale;pred=K@coef;rows[f's{source}_t{target}_error']=(pred-targets[target]).square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
  print('source',source,'complete',flush=True)
 matrix=np.array([[1-rows[f's{s}_t{t}_error'].sum()/rows[f't{t}_energy'].sum() for t in range(18)] for s in range(18)]);old=np.load(ART/'bilin18_l17_kernel_confirmation_rows.npz');replay=max(float(np.max(np.abs(rows[f's{g}_t{g}_error']-old[f'g{g}_interaction_error']))/np.max(np.abs(old[f'g{g}_interaction_error']))) for g in range(18));assert replay<1e-6;summary={}
 for family,group in [('actual',list(range(9))),('angle',list(range(9,18)))]:
  energy=sum(rows[f't{t}_energy'].sum() for t in group);same=sum(rows[f's{t}_t{t}_error'].sum() for t in group);other=sum(np.mean([rows[f's{s}_t{t}_error'].sum() for s in group if s!=t]) for t in group);cross=sum(rows[f's{(t+9)%18}_t{t}_error'].sum() for t in group);summary[family]={'same_group_fidelity':float(1-same/energy),'other_head_mean_fidelity':float(1-other/energy),'same_head_other_payload_fidelity':float(1-cross/energy),'other_head_retention':float((1-other/energy)/(1-same/energy))}
 gates={'other_head80retention':all(v['other_head_retention']>=.8 for v in summary.values()),'cross_payload80retention':all(v['same_head_other_payload_fidelity']>=.8*v['same_group_fidelity'] for v in summary.values())};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'matrix':matrix.tolist(),'summary':summary,'diagonal_replay_relative':replay,'gates':gates},indent=2)+'\n');print('FINISHED',summary,gates,'replay',replay,flush=True)
if __name__=='__main__':main()
