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
STEM=ART/'bilin18_l17_kernel_support'
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
 rows={};audits=[]
 for target,(w,t) in enumerate(features):
  diag=w.square().sum(1)*t.square().sum(1)+u.square().sum(1)*v.square().sum(1)-2*(u*w).sum(1)*(v*t).sum(1);assert float(diag.min())>0;rows[f't{target}_feature_energy']=diag.reshape(128,4).sum(1).cpu().numpy()
 for source in range(19):
  if source<18:
   f=np.load(folder/f'g{source}.npz');ref=(ux,vx,torch.tensor(f['uz'],device='cuda'),torch.tensor(f['vz'],device='cuda'));name='specialist';tests=[source]
  else:
   f=np.load(ART/'bilin18_l17_shared_kernel_mixed_frozen.npz');ref=tuple(torch.tensor(f[k],device='cuda') for k in ['ux','vx','uz','vz']);name='mixed';tests=range(18)
  K=kern(ref,ref);ev,U=torch.linalg.eigh((K+K.T)/2);audit={'source':source,'max_eigenvalue':float(ev.max()),'min_eigenvalue':float(ev.min()),'ranks':{}}
  for cutoff in cfg['cutoffs']:audit['ranks'][str(cutoff)]=int((ev>ev.max()*cutoff).sum())
  for target in tests:
   w,t=features[target];cross=kern((u,v,w,t),ref);left=cross@U;diag=w.square().sum(1)*t.square().sum(1)+u.square().sum(1)*v.square().sum(1)-2*(u*w).sum(1)*(v*t).sum(1)
   for cutoff in cfg['cutoffs']:
    keep=ev>ev.max()*cutoff;captured=(left[:,keep].square()/ev[keep]).sum(1);fraction=captured/diag;assert float(fraction.min())>=-1e-6 and float(fraction.max())<=1+1e-6;rows[f'{name}_t{target}_c{cutoff}_captured']=captured.reshape(128,4).sum(1).cpu().numpy();rows[f'{name}_t{target}_c{cutoff}_fraction']=fraction.reshape(128,4).mean(1).cpu().numpy()
  audits.append(audit);print('source',source,'complete',flush=True)
 summary={}
 for family,group in [('actual',range(9)),('angle',range(9,18))]:
  energy=sum(rows[f't{t}_feature_energy'].sum() for t in group);summary[family]={}
  for name in ['specialist','mixed']:
   summary[family][name]={str(cutoff):{'feature_energy_coverage':float(sum(rows[f'{name}_t{t}_c{cutoff}_captured'].sum() for t in group)/energy),'equal_site_coverage':float(np.mean([rows[f'{name}_t{t}_c{cutoff}_fraction'] for t in group]))} for cutoff in cfg['cutoffs']}
 primary=str(cfg['cutoffs'][0]);gates={'mixed50':all(v['mixed'][primary]['feature_energy_coverage']>=.5 for v in summary.values()),'specialist_gap20':all(v['specialist'][primary]['feature_energy_coverage']-v['mixed'][primary]['feature_energy_coverage']>=.2 for v in summary.values())};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'spectral_audits':audits,'gates':gates},indent=2)+'\n');print('FINISHED',summary,gates,flush=True)
if __name__=='__main__':main()
