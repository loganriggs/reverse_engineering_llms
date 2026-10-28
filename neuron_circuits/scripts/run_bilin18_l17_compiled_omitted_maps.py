"""H439 exact fixed-direction compilation and weight-only output compression."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_compiled_omitted_maps'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 z=np.load(ART/'bilin18_l17_operator_transfer_confirmation_rows.npz');model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach().double() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q.double();m=np.load(ART/'bilin18_l17_unmatched_moment_controls_moments.npz');mu=torch.tensor(m['mean'][1],device='cuda');cov=torch.tensor(m['covariance'][1],device='cuda');ev,U=torch.linalg.eigh(cov+mu[:,None]*mu[None,:]);root=(U*ev.sqrt())@U.T;inv=(U*ev.rsqrt())@U.T;V=torch.tensor(np.load(ART/'bilin18_l17_omitted_operator_shared_frozen.npz')['basis'][:,:16],device='cuda');H=Q@root@V;sets=np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz');ix=torch.tensor(sets['sets128'][0],device='cuda');keep=torch.ones(4608,dtype=torch.bool,device='cuda');keep[ix]=False;Lo,Ro,Do=L[keep],R[keep],D[:,keep];lh=Lo@H;rh=Ro@H;maps=[]
 for k in range(16):
  T=Do@(lh[:,k,None]*Ro+rh[:,k,None]*Lo);maps.append(T);print('compiled map',k,'complete',flush=True)
 maps=torch.stack(maps);gram=torch.einsum('roi,rpi->op',maps,maps);evals,B=torch.linalg.eigh(gram);B=B.flip(-1);np.savez_compressed(STEM.with_name(STEM.name+'_weight_compilation.npz'),source_directions=H.cpu().numpy(),output_basis=B.cpu().numpy(),output_eigenvalues=evals.flip(-1).cpu().numpy(),maps=maps.cpu().numpy());print('WEIGHT BASIS FROZEN',flush=True)
 pre=torch.tensor(z['pre'],device='cuda').double();st=torch.tensor(z['state'],device='cuda');base=F.rms_norm(st,(1152,));s=(pre.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pre/s;mo=((x@Lo.T)*(x@Ro.T))@Do.T;A=torch.einsum('roi,ni->nro',maps,x);radial=2*(x@H/1152)[:,:,None]*mo[:,None,:];writes=-(A-radial)/s[:,:,None];src=torch.tensor(z['sources'],device='cuda').double();coeff=(src@Q)@inv@V;compiled=torch.einsum('ar,nro->ano',coeff,writes);ref=torch.tensor(z['predictions'][3],device='cuda');replay=float(((compiled-ref).norm(dim=-1)/ref.norm(dim=-1).clamp_min(1e-12)).max());full=torch.tensor(z['full'],device='cuda').double();ret=torch.tensor(z['retained'],device='cuda').double();predictions=[compiled];labels=['exact_compiled'];rankcaptures=[]
 for rank in cfg['ranks']:
  b=B[:,:rank];consistent=(compiled@b)@b.T;only_linear=torch.einsum('ar,nro->ano',coeff,-((A@b)@b.T-radial)/s[:,:,None]);predictions.extend([consistent,only_linear]);labels.extend([f'project_both_{rank}',f'project_linear_only_{rank}']);rankcaptures.append(float(evals.flip(-1)[:rank].sum()/evals.sum()))
 predictions=torch.stack(predictions);errors=np.empty((len(labels),24,32,2));rawerrors=(full[None]-ret[None]-predictions).square().sum(-1).cpu().numpy()
 for si in range(32):
  ss=st[si:si+1];th=torch.tanh(base[si:si+1].double()@W.T/30);p=(30*th).softmax(-1);cap=1-th.square()
  def tangent(v):
   dt=(native_drms(ss.double(),v)@W.T)*cap;return dt-(p*dt).sum(-1,keepdim=True)
  def logprob(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  tf=tangent(full[:,si]);lt=logprob(full[:,si]);pt=lt.exp()
  for method in range(len(labels)):
   v=ret[:,si]+predictions[method,:,si];res=tf-tangent(v);lp=logprob(v);errors[method,:,si]=torch.stack([.5*(p*res.square()).sum(-1),(pt*(lt-lp)).sum(-1)],-1).cpu().numpy()
  print('evaluation',si,'complete',flush=True)
 assert np.isfinite(errors).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,raw_errors=rawerrors,predictions=predictions.cpu().numpy());j={'experiment':'H439','config':cfg,'methods':labels,'weight_energy_capture':rankcaptures,'compiled_prediction_replay_relative_max':replay,'compiled_replay_gate':replay<1e-9,'unique_completed_documents':12480};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',replay,flush=True)
from run_bilin18_l17_fisher_ranked_factors import native_drms
if __name__=='__main__':main()
