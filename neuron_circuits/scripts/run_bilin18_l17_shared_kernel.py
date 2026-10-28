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
STEM=ART/'bilin18_l17_shared_kernel'
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
 path=ART/'bilin18_l17_document_repartition_responses.npy';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['response_sha256'];Y=torch.tensor(np.load(path),device='cuda').reshape(18,4096,1152)[:,:3072].double();refs=[]
 for g in range(18):
  f=np.load(folder/f'g{g}.npz');refs.append((torch.tensor(f['uz'],device='cuda'),torch.tensor(f['vz'],device='cuda')))
 rng=np.random.default_rng(cfg['assignment_seed']);assignments={};audits=[]
 for name,groups in [('actual',np.arange(9)),('angle',np.arange(9,18)),('mixed',np.arange(18))]:
  assign=np.resize(groups,3072);rng.shuffle(assign);assignments[name]=assign.tolist();ag=torch.tensor(assign,device='cuda');site=torch.arange(3072,device='cuda');wref=torch.stack([refs[int(g)][0][i] for i,g in enumerate(assign)]);tref=torch.stack([refs[int(g)][1][i] for i,g in enumerate(assign)]);yy=Y[ag,site];K=(wref@wref.T)*(tref@tref.T)+(ux@ux.T)*(vx@vx.T)-(wref@ux.T)*(tref@vx.T)-(ux@wref.T)*(vx@tref.T);scale=K.diag().mean();K=K/scale;system=(K+K.T)/2+cfg['ridge']*torch.eye(3072,device='cuda',dtype=torch.float64);coef=torch.cholesky_solve(yy,torch.linalg.cholesky(system));residual=float((system@coef-yy).norm()/yy.norm());assert residual<1e-8;audits.append({'name':name,'solve_relative':residual})
  for target,(w,t) in enumerate(features):
   Kt=((w@wref.T)*(t@tref.T)+(u@ux.T)*(v@vx.T)-(w@ux.T)*(t@vx.T)-(u@wref.T)*(v@tref.T))/scale;pred=Kt@coef;rows[f'{name}_t{target}_error']=(pred-targets[target]).square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_'+name+'_frozen.npz'),ux=ux.cpu().numpy(),vx=vx.cpu().numpy(),uz=wref.cpu().numpy(),vz=tref.cpu().numpy(),coef=coef.cpu().numpy(),kernel_scale=np.array(float(scale)),assignment=assign);print('fit',name,'complete',flush=True)
 summary={};old=json.loads((ART/'bilin18_l17_kernel_confirmation.json').read_text())['summary'];boot=np.random.default_rng(20270215).integers(128,size=(2000,128))
 for family,group in [('actual',range(9)),('angle',range(9,18))]:
  energy=sum(rows[f't{t}_energy'] for t in group);summary[family]={}
  for name in ['actual','angle','mixed']:
   err=sum(rows[f'{name}_t{t}_error'] for t in group);summary[family][name]={'fidelity':float(1-err.sum()/energy.sum()),'paired95':np.quantile(1-err[boot].sum(1)/energy[boot].sum(1),[.025,.975]).tolist(),'per_head':[float(1-rows[f'{name}_t{t}_error'].sum()/rows[f't{t}_energy'].sum()) for t in group]}
 gates={'mixed10':all(v['mixed']['fidelity']>=.1 for v in summary.values()),'mixed80retention':all(v['mixed']['fidelity']>=.8*old[f]['interaction']['fidelity'] for f,v in summary.items())};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'assignments':assignments,'summary':summary,'solve_audits':audits,'gates':gates},indent=2)+'\n');print('FINISHED',summary,gates,flush=True)
if __name__=='__main__':main()
