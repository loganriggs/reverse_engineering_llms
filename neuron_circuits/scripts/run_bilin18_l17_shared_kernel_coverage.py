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
STEM=ART/'bilin18_l17_shared_kernel_coverage'
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
 def kern(a,b):
  aa,bb,cc,dd=a;ee,ff,gg,hh=b;return (cc@gg.T)*(dd@hh.T)+(aa@ee.T)*(bb@ff.T)-(cc@ee.T)*(dd@ff.T)-(aa@gg.T)*(bb@hh.T)
 path=ART/'bilin18_l17_document_repartition_responses.npy';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['response_sha256'];Y=torch.tensor(np.load(path),device='cuda').reshape(18,4096,1152)[:,:3072].double();path=ART/'bilin18_l17_shared_kernel_mixed_frozen.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['mixed_sha256'];f=np.load(path);assignment=f['assignment'];second=(assignment+cfg['group_offset'])%18;assert np.all(second!=assignment);scale=float(f['kernel_scale']);refs=[]
 for g in range(18):
  z=np.load(folder/f'g{g}.npz');refs.append((torch.tensor(z['uz'],device='cuda'),torch.tensor(z['vz'],device='cuda')))
 uz2=torch.stack([refs[int(g)][0][i] for i,g in enumerate(second)]);vz2=torch.stack([refs[int(g)][1][i] for i,g in enumerate(second)]);refall=(torch.cat([ux,ux]),torch.cat([vx,vx]),torch.cat([torch.tensor(f['uz'],device='cuda'),uz2]),torch.cat([torch.tensor(f['vz'],device='cuda'),vz2]));site=torch.arange(3072,device='cuda');labels=torch.cat([Y[torch.tensor(assignment,device='cuda'),site],Y[torch.tensor(second,device='cuda'),site]]);rows={};audit=[]
 for n in cfg['budgets']:
  ref=tuple(vv[:n] for vv in refall);K=kern(ref,ref)/scale;ev,U=torch.linalg.eigh((K+K.T)/2);keep=ev>ev.max()*cfg['cutoff'];rhs=U.T@labels[:n];alpha=cfg['base_ridge']*n/3072;coef=U@(rhs/(ev+alpha)[:,None]);audit.append({'n':n,'rank':int(keep.sum()),'min_eigenvalue':float(ev.min()),'ridge':alpha,'solve_relative':float(((K+alpha*torch.eye(n,device='cuda',dtype=torch.float64))@coef-labels[:n]).norm()/labels[:n].norm())})
  for target,(w,t) in enumerate(features):
   left=(kern((u,v,w,t),ref)/scale)@U;covered=(left[:,keep]/ev[keep][None,:])@rhs[keep];uncovered=targets[target]-covered;pred=(left/(ev+alpha)[None,:])@rhs;diagonal=(w.square().sum(1)*t.square().sum(1)+u.square().sum(1)*v.square().sum(1)-2*(u*w).sum(1)*(v*t).sum(1))/scale;captured=(left[:,keep].square()/ev[keep]).sum(1)
   vals={'feature_energy':diagonal,'feature_captured':captured,'target_energy':targets[target].square().sum(1),'ridge_error':(pred-targets[target]).square().sum(1),'covered_energy':covered.square().sum(1),'uncovered_energy':uncovered.square().sum(1),'dot':(covered*uncovered).sum(1)}
   for key,val in vals.items():rows[f'n{n}_t{target}_{key}']=val.reshape(128,4).sum(1).cpu().numpy()
  if n==6144:np.savez_compressed(STEM.with_name(STEM.name+'_6144_frozen.npz'),**{k:vv.cpu().numpy() for k,vv in zip(['ux','vx','uz','vz'],ref)},coef=coef.cpu().numpy(),kernel_scale=np.array(scale),ridge=np.array(alpha),assignment=np.r_[assignment,second])
  print('budget',n,'complete',flush=True)
 summary={}
 for family,group in [('actual',range(9)),('angle',range(9,18))]:
  summary[family]={}
  for n in cfg['budgets']:
   sums={k:sum(rows[f'n{n}_t{t}_{k}'].sum() for t in group) for k in vals};summary[family][str(n)]={'feature_coverage':float(sums['feature_captured']/sums['feature_energy']),'ridge_fidelity':float(1-sums['ridge_error']/sums['target_energy']),'projection_fidelity':float(1-sums['uncovered_energy']/sums['target_energy']),'component_cosine':float(sums['dot']/np.sqrt(sums['covered_energy']*sums['uncovered_energy']))}
 gates={'coverage_gain10':all(v['6144']['feature_coverage']-v['3072']['feature_coverage']>=.1 for v in summary.values()),'response_gain5':all(v['6144']['ridge_fidelity']-v['3072']['ridge_fidelity']>=.05 for v in summary.values()),'response10':all(v['6144']['ridge_fidelity']>=.1 for v in summary.values())};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'audits':audit,'gates':gates},indent=2)+'\n');print('FINISHED',summary,gates,flush=True)
if __name__=='__main__':main()
