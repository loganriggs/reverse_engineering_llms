"""H434 independent baseline-selected factor transfer."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_baseline_context_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();allpre,_,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);xx=F.rms_norm(allpre,(1152,));allstate=allpre+((xx@L.T)*(xx@R.T))@D.T+model.blocks[17].mlp.Down_bias;rms=allstate.double().square().mean(-1).sqrt().cpu().numpy();low=np.argsort(rms,kind='stable')[:16];mid=np.array([i for i in np.argsort(abs(rms-np.median(rms)),kind='stable') if i not in low])[:16];sites=np.r_[low,mid];selection={'low':low.tolist(),'median':mid.tolist(),'all_output_rms':rms.tolist(),'selection_rule':'16 lowest RMS and16 nearest median, disjoint, stable index tie-break; no intervention scores'};STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps(selection,indent=2)+'\n');print('SELECTION FROZEN',sites,flush=True)
 pre=allpre[sites];st=allstate[sites];base=F.rms_norm(st,(1152,));_,dh,_=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),cfg);Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q.double();m=np.load(ART/'bilin18_l17_unmatched_moment_controls_moments.npz');mu=torch.tensor(m['mean'][1],device='cuda');root=torch.tensor(m['root'][1],device='cuda');src=[];donor_ids=[]
 for seed in cfg['source_seeds']:
  di=int(np.random.default_rng(seed).integers(512));donor_ids.append(di);src.append(dh[di,6].double())
 for seed in cfg['source_seeds']:
  noise=torch.tensor(np.random.default_rng(seed+100000).standard_normal(128),device='cuda');src.append((mu+noise@root.T)@Q.T)
 src=torch.stack(src);sets=np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz');ids=[torch.tensor(v,device='cuda') for v in [sets['core'],sets['sets128'][0,64:],sets['sets128'][0]]];x=F.rms_norm(pre,(1152,));phi=(x@L.T)*(x@R.T);parts=[];full=[]
 for h in src:
  xx=F.rms_norm(pre-h.float(),(1152,));p1=(xx@L.T)*(xx@R.T);delta=p1-phi;parts.append(torch.stack([delta[:,ix]@D[:,ix].T for ix in ids],1));full.append(p1@D.T-phi@D.T)
 parts=torch.stack(parts);full=torch.stack(full);norm=full.double().norm(dim=-1);common=norm.flatten().median();partition=float(((parts[:,:,0]+parts[:,:,1]-parts[:,:,2]).norm(dim=-1)/(parts[:,:,2].norm(dim=-1)+1e-12)).max());errors=np.empty((2,12,32,32,3,2));effects=np.empty((2,12,32,32,2));rawerror=(full[:,:,None]-parts).double().square().sum(-1).cpu().numpy();raweffect=full.double().square().sum(-1).cpu().numpy()
 for dst in range(32):
  ss=st[dst:dst+1];bb=base[dst:dst+1];th=torch.tanh(bb.double()@W.T/30);cap=1-th.square();lp=(30*th).log_softmax(-1);p=lp.exp()
  def tangent(v):
   dt=(native_drms(ss.double(),v.double())@W.T)*cap;return dt-(p*dt).sum(-1,keepdim=True)
  def logprob(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  for ni in range(2):
   scale=torch.ones_like(norm) if ni==0 else common/norm;ff=(full.double()*scale[:,:,None]).reshape(-1,1152);pp=(parts.double()*scale[:,:,None,None]).reshape(-1,3,1152)
   for off in range(0,len(ff),32):
    f=ff[off:off+32];t=tangent(f);lt=logprob(f);pt=lt.exp();ef=torch.stack([.5*(p*t.square()).sum(-1),(pt*(lt-lp)).sum(-1)],-1).cpu().numpy();effects[ni,:,:,dst].reshape(-1,2)[off:off+len(f)]=ef
    for gi in range(3):
     v=pp[off:off+32,gi];res=t-tangent(v);lg=logprob(v);er=torch.stack([.5*(p*res.square()).sum(-1),(pt*(lt-lg)).sum(-1)],-1).cpu().numpy();errors[ni,:,:,dst,gi].reshape(-1,2)[off:off+len(f)]=er
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:,:,:dst+1],effects=effects[:,:,:,:dst+1]);print('destination',dst,'complete',flush=True)
 assert np.isfinite(errors).all() and np.isfinite(effects).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,effects=effects,raw_errors=rawerror,raw_effects=raweffect,parts=parts.cpu().numpy(),full=full.cpu().numpy(),sources=src.cpu().numpy(),pre=pre.cpu().numpy(),state=st.cpu().numpy());j={'experiment':'H434','config':cfg,'selection':selection,'donor_ids':donor_ids,'common_response_norm':float(common),'partition_relative_error':partition,'numerical_gates':{'partition':partition<cfg['partition_tolerance'],'nonnegative_KL':bool(errors[...,1].min()>-1e-10 and effects[...,1].min()>-1e-10)},'unique_completed_documents':11968};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()
