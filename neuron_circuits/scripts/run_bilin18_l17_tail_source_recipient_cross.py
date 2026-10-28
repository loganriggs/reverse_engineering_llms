import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_tail_source_recipient_cross'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());old=json.loads((ROOT/'configs/bilin18_l17_centered_source_reflection.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];Ld,Rd,Dd=L.double(),R.double(),D.double();W=model.lm_head.weight.double();Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q.double();_,donor,_=collect_sources(model,np.load(ROOT/old['donor_corpus']),dict(old,rows=512));mom=np.load(ART/'bilin18_l17_unmatched_moment_controls_moments.npz');mu=torch.tensor(mom['mean'],device='cuda');root=torch.tensor(mom['root'],device='cuda');mean=mu[1]@Q.T;sources=[]
 for family in range(2):
  for di,seed in enumerate(old['draw_seeds']):
   if family==0:
    doc=np.random.default_rng(seed).permutation(512)[199];h=donor[doc*4+1,6].double()
   else:
    noise=torch.tensor(np.random.default_rng(seed+100000).standard_normal((512,4,128)),device='cuda');cc=mu+torch.einsum('npi,pji->npj',noise,root);h=cc[199,1]@Q.T
   sources.extend([h,2*mean-h])
 sources=torch.stack(sources);common=sources.norm(dim=-1).median();states=np.load(ART/'bilin18_l17_recipient_tail_diagnostic_current_states.npz');sites=cfg['sites'];pre=torch.tensor(states['pre'][sites],device='cuda');st=torch.tensor(states['state'][sites],device='cuda');base=torch.tensor(states['base'][sites],device='cuda');pd=pre.double();x=pd/(pd.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();lx=x@Ld.T;rx=x@Rd.T;xf=F.rms_norm(pre,(1152,));phi=(xf@L.T)*(xf@R.T);native0=phi@D.T;th=torch.tanh(base.double()@W.T/30);cap=1-th.square();lp=(30*th).log_softmax(-1);p=lp.exp()
 def tangent(v):
  dt=(native_drms(st.double(),v)@W.T)*cap;return dt-(p*dt).sum(-1,keepdim=True)
 # source, normcontrol, response mode(native/linearRMS), amplitude, recipient, metric(norm2,Fisher,KL)
 vals=np.empty((48,2,2,3,17,3));quartic=np.empty((48,2,17));reference=np.load(ART/'bilin18_l17_centered_source_reflection_rows.npz')['coefficients'];replay=0.
 for si,h0 in enumerate(sources):
  for ni in range(2):
   h=h0 if ni==0 else h0*common/h0.norm();du=native_drms(pd,-h.expand_as(pd));ul=du@Ld.T;ur=du@Rd.T;b=(ul*ur)@Dd.T;tb=tangent(b);qe=.5*(p*tb.square()).sum(-1);quartic[si,ni]=qe.cpu()
   if ni==0:
    fam=si//24;di=(si%24)//2;ri=si%2;ref=reference[fam,di,ri,2,0,797];replay=max(replay,abs(float(qe[0])-ref)/max(abs(ref),1e-20))
   for ai,amp in enumerate(cfg['amplitudes']):
    changed=F.rms_norm(pre-amp*h.float(),(1152,));native=((changed@L.T)*(changed@R.T))@D.T-native0;u=amp*du;linear=(((x+u)@Ld.T)*((x+u)@Rd.T)-lx*rx)@Dd.T
    for mode,response in enumerate([native,linear]):
     v=response.double();t=tangent(v);target=F.rms_norm(st+response.float(),(1152,));lt=(30*torch.tanh(target.double()@W.T/30)).log_softmax(-1);kl=(lt.exp()*(lt-lp)).sum(-1);vals[si,ni,mode,ai]=torch.stack([v.square().sum(-1),.5*(p*t.square()).sum(-1),kl],-1).cpu()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),values=vals[:si+1],quartic=quartic[:si+1]);print('source',si,'complete',flush=True)
 assert np.isfinite(vals).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),values=vals,quartic=quartic,sources=sources.cpu().numpy());means=vals.mean(0);ratios=means[:,:,:,0]/means[:,:,:,1:].mean(3);j={'experiment':'H431','config':cfg,'source_norms':sources.norm(dim=-1).cpu().tolist(),'common_source_norm':float(common),'tail_over_control_mean_effect':ratios.tolist(),'metric_order':['raw_response_norm_squared','Fisher_energy','exact_KL'],'quartic_tail_over_control_mean':(quartic.mean(0)[:,0]/quartic.mean(0)[:,1:].mean(1)).tolist(),'quartic_replay_relative_max':replay,'quartic_replay_gate':bool(replay<cfg['numerical_tolerance_quartic_replay_relative']),'unique_completed_documents':11456};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',replay,ratios,flush=True)
if __name__=='__main__':main()
