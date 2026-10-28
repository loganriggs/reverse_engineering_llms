"""H433 frozen factors across origins and output states; post-selected diagnostic."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_factor_output_transfer'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 states=np.load(ART/'bilin18_l17_recipient_tail_diagnostic_current_states.npz');old=np.load(ART/'bilin18_l17_output_metric_cross_rows.npz');src=np.load(ART/'bilin18_l17_tail_source_recipient_cross_rows.npz')['sources'];sets=np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz');model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();ids=[torch.tensor(v,device='cuda') for v in [sets['core'],sets['sets128'][0,64:],sets['sets128'][0]]];pre=torch.tensor(states['pre'][cfg['sites']],device='cuda');st=torch.tensor(states['state'][cfg['sites']],device='cuda');base=torch.tensor(states['base'][cfg['sites']],device='cuda');x=F.rms_norm(pre,(1152,));phi=(x@L.T)*(x@R.T);parts=[]
 for h in src:
  xx=F.rms_norm(pre-torch.tensor(h,device='cuda').float(),(1152,));delta=(xx@L.T)*(xx@R.T)-phi;parts.append(torch.stack([delta[:,ix]@D[:,ix].T for ix in ids],1))
 parts=torch.stack(parts);full=torch.tensor(old['responses'],device='cuda');norm=full.double().norm(dim=-1);common=norm.flatten().median();partition=float(((parts[:,:,0]+parts[:,:,1]-parts[:,:,2]).norm(dim=-1)/(parts[:,:,2].norm(dim=-1)+1e-12)).max());errors=np.empty((2,48,17,17,3,2));effects=np.empty((2,48,17,17,2));rawerror=(full[:,:,None]-parts).double().square().sum(-1).cpu().numpy();raweffect=full.double().square().sum(-1).cpu().numpy()
 for dst in range(17):
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
 assert np.isfinite(errors).all() and np.isfinite(effects).all();replay=float(np.max(abs(effects-old['values'])/np.maximum(abs(old['values']),1e-12)));np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,effects=effects,raw_errors=rawerror,raw_effects=raweffect,parts=parts.cpu().numpy());j={'experiment':'H433','config':cfg,'partition_relative_error':partition,'effect_replay_relative_error':replay,'numerical_gates':{'partition':partition<cfg['partition_tolerance'],'effect_replay':replay<cfg['replay_tolerance']},'unique_completed_documents':11456};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],partition,replay,flush=True)
if __name__=='__main__':main()
