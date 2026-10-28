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
STEM=ART/'bilin18_l17_kernel_output_decomposition'
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
 path=ART/'bilin18_l17_document_repartition_responses.npy';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['response_sha256'];Y=torch.tensor(np.load(path),device='cuda').reshape(18,4096,1152)[:,:3072].double();path=ART/'bilin18_l17_shared_kernel_mixed_frozen.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['mixed_sha256'];rows={};audits=[]
 for source in range(19):
  if source<18:
   f=np.load(folder/f'g{source}.npz');ref=(ux,vx,torch.tensor(f['uz'],device='cuda'),torch.tensor(f['vz'],device='cuda'));labels=Y[source];name='specialist';tests=[source]
  else:
   f=np.load(path);ref=tuple(torch.tensor(f[k],device='cuda') for k in ['ux','vx','uz','vz']);labels=Y[torch.tensor(f['assignment'],device='cuda'),torch.arange(3072,device='cuda')];name='mixed';tests=range(18)
  K=kern(ref,ref);ev,U=torch.linalg.eigh((K+K.T)/2);rhs=U.T@labels;scale=K.diag().mean()
  for target in tests:
   w,t=features[target];left=kern((u,v,w,t),ref)@U;truth=targets[target]
   for cutoff in [1e-10,1e-6]:
    keep=ev>ev.max()*cutoff;covered=(left[:,keep]/ev[keep][None,:])@rhs[keep];uncovered=truth-covered;prefix=f'{name}_t{target}_c{cutoff}'
    for metric,value in [('target',truth.square().sum(1)),('covered',covered.square().sum(1)),('uncovered',uncovered.square().sum(1)),('dot',(covered*uncovered).sum(1))]:rows[prefix+'_'+metric]=value.reshape(128,4).sum(1).cpu().numpy()
   for alpha in [.01,10.]:
    pred=(left/(ev+alpha*scale)[None,:])@rhs;rows[f'{name}_t{target}_ridge{alpha}_error']=(pred-truth).square().sum(1).reshape(128,4).sum(1).cpu().numpy()
  keep=ev>ev.max()*1e-10;coef=U[:,keep]@(rhs[keep]/ev[keep,None]);audits.append({'source':source,'retained_rank':int(keep.sum()),'training_projection_relative':float((K@coef-labels).norm()/labels.norm())});print('source',source,'complete',flush=True)
 summary={}
 for family,group in [('actual',range(9)),('angle',range(9,18))]:
  summary[family]={}
  for name in ['specialist','mixed']:
   summary[family][name]={}
   for cutoff in [1e-10,1e-6]:
    totals={m:sum(rows[f'{name}_t{t}_c{cutoff}_{m}'].sum() for t in group) for m in ['target','covered','uncovered','dot']};E=totals['target'];summary[family][name][str(cutoff)]={'covered_energy_ratio':float(totals['covered']/E),'uncovered_energy_ratio':float(totals['uncovered']/E),'twice_dot_ratio':float(2*totals['dot']/E),'component_cosine':float(totals['dot']/np.sqrt(totals['covered']*totals['uncovered'])),'projection_fidelity':float(1-totals['uncovered']/E),'identity_relative_error':float(abs(E-totals['covered']-totals['uncovered']-2*totals['dot'])/E),'ridge_fidelity':{str(alpha):float(1-sum(rows[f'{name}_t{t}_ridge{alpha}_error'].sum() for t in group)/E) for alpha in [.01,10.]}}
 gates={'mixed_cancellation':all(v['mixed']['1e-10']['component_cosine']<=-.1 for v in summary.values()),'mixed_uncovered_dominates':all(v['mixed']['1e-10']['uncovered_energy_ratio']>=.9 for v in summary.values())};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'training_audits':audits,'gates':gates},indent=2)+'\n');print('FINISHED',summary,gates,flush=True)
if __name__=='__main__':main()
