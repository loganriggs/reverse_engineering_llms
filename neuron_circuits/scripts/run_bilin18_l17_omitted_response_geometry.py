"""H435 post-primary omitted-response geometry on the H434 panel."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_omitted_response_geometry'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());path=ART/'bilin18_l17_baseline_context_confirmation_rows.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['reference_sha256'];z=np.load(path);full=torch.tensor(z['full'],device='cuda').double();ret=torch.tensor(z['parts'][:,:,2],device='cuda').double();omit=full-ret;alpha=(omit*full).sum(-1)/full.square().sum(-1);par=alpha[:,:,None]*full;orth=omit-par;raw=torch.stack([full.square().sum(-1),ret.square().sum(-1),omit.square().sum(-1),2*(ret*omit).sum(-1),par.square().sum(-1),orth.square().sum(-1),2*(par*orth).sum(-1)],-1).cpu().numpy();st=torch.tensor(z['state'],device='cuda');base=F.rms_norm(st,(1152,));model=load_full_model('cuda');model.lm_head.float();W=model.lm_head.weight.double();energies=np.empty((12,32,32,7))
 for dst in range(32):
  ss=st[dst:dst+1].double();th=torch.tanh(base[dst:dst+1].double()@W.T/30);cap=1-th.square();p=(30*th).softmax(-1)
  def tangent(v):
   dt=(native_drms(ss,v)@W.T)*cap;return dt-(p*dt).sum(-1,keepdim=True)
  ff=full.reshape(-1,1152);rr=ret.reshape(-1,1152);aa=alpha.reshape(-1)
  for off in range(0,len(ff),32):
   tf=tangent(ff[off:off+32]);tr=tangent(rr[off:off+32]);to=tf-tr;tp=aa[off:off+32,None]*tf;tt=to-tp;energy=lambda t:.5*(p*t.square()).sum(-1);cross=lambda u,v:(p*u*v).sum(-1);arr=torch.stack([energy(tf),energy(tr),energy(to),cross(tr,to),energy(tp),energy(tt),cross(tp,tt)],-1).cpu().numpy();energies[:,:,dst].reshape(-1,7)[off:off+len(arr)]=arr
  print('destination',dst,'complete',flush=True)
 assert np.isfinite(energies).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),energies=energies,raw=raw,alpha=alpha.cpu().numpy());refe=z['effects'][0,:,:,:,0];refo=z['errors'][0,:,:,:,2,0];replay=max(float(np.max(abs(energies[...,0]-refe)/np.maximum(abs(refe),1e-20))),float(np.max(abs(energies[...,2]-refo)/np.maximum(abs(refo),1e-20))));closure=max(float(np.max(abs(energies[...,0]-energies[...,1]-energies[...,2]-energies[...,3])/np.maximum(energies[...,0]+energies[...,1]+energies[...,2],1e-20))),float(np.max(abs(energies[...,2]-energies[...,4]-energies[...,5]-energies[...,6])/np.maximum(energies[...,2]+energies[...,4]+energies[...,5],1e-20))));j={'experiment':'H435','config':cfg,'channels':['full','retained_union','omitted','retained_omitted_cross','raw_parallel_omitted','raw_orthogonal_omitted','parallel_orthogonal_cross'],'replay_relative_max':replay,'closure_relative_max':closure,'gates':{'replay':replay<1e-7,'closure':closure<1e-10},'unique_completed_documents':11968};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['gates'],replay,closure,flush=True)
if __name__=='__main__':main()
