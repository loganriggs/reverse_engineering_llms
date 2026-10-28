"""H436 input-derived omitted-response Taylor predictors, not compression."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_omitted_local_prediction'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());path=ART/'bilin18_l17_baseline_context_confirmation_rows.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['reference_sha256'];z=np.load(path);model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach().double() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();pre=torch.tensor(z['pre'],device='cuda').double();st=torch.tensor(z['state'],device='cuda');base=F.rms_norm(st,(1152,));src=torch.tensor(z['sources'],device='cuda').double();full=torch.tensor(z['full'],device='cuda').double();ret=torch.tensor(z['parts'][:,:,2],device='cuda').double();sets=np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz');assert hashlib.sha256((ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz').read_bytes()).hexdigest()==cfg['sets_sha256'];keep=torch.ones(4608,dtype=torch.bool,device='cuda');keep[torch.tensor(sets['sets128'][0],device='cuda')]=False;s=(pre.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pre/s;lx=x@L.T;rx=x@R.T;pred=[]
 for h in src:
  v=-h.expand_as(pre);m=(pre*v).mean(-1,keepdim=True);v2=v.square().mean(-1,keepdim=True);u=native_drms(pre,v);u2=pre*(-v2/(2*s**3)+3*m*m/(2*s**5))-v*m/s**3;ul=u@L.T;ur=u@R.T;first=ul*rx+lx*ur;second=(u2@L.T)*rx+lx*(u2@R.T)+ul*ur;g1=first[:,keep]@D[:,keep].T;g2=second[:,keep]@D[:,keep].T;pred.append(torch.stack([g1,g1+g2]))
 pred=torch.stack(pred,1);wrong=pred[0,torch.tensor(np.r_[np.roll(np.arange(6),1),6+np.roll(np.arange(6),1)],device='cuda')];pred=torch.cat([pred,wrong[None]],0);corrected=ret[None]+pred;rawerr=(full[None]-corrected).square().sum(-1).cpu().numpy();errors=np.empty((3,12,32,32,2));audit=0.
 for dst in range(32):
  ss=st[dst:dst+1];th=torch.tanh(base[dst:dst+1].double()@W.T/30);cap=1-th.square();lp=(30*th).log_softmax(-1);p=lp.exp()
  def tangent(v):
   dt=(native_drms(ss.double(),v)@W.T)*cap;return dt-(p*dt).sum(-1,keepdim=True)
  def logprob(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  ff=full.reshape(-1,1152)
  for off in range(0,len(ff),32):
   f=ff[off:off+32];tf=tangent(f);lt=logprob(f);pt=lt.exp()
   for method in range(3):
    vv=corrected[method].reshape(-1,1152)[off:off+len(f)];res=tf-tangent(vv);lg=logprob(vv);val=torch.stack([.5*(p*res.square()).sum(-1),(pt*(lt-lg)).sum(-1)],-1).cpu().numpy();errors[method,:,:,dst].reshape(-1,2)[off:off+len(f)]=val
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:,:,:dst+1]);print('destination',dst,'complete',flush=True)
 assert np.isfinite(errors).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,raw_errors=rawerr,predictions=pred.cpu().numpy());j={'experiment':'H436','config':cfg,'methods':['linear_omitted','second_order_omitted','wrong_source_linear'],'unique_completed_documents':11968,'cost_note':'Computes both learned input reads for all4608 factors and projects4480 omitted factors. Analytic, target-free prediction but dense machinery; not a compact unit or acceleration.'};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',flush=True)
if __name__=='__main__':main()
