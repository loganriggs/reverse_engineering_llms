import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_source_mean_variation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 old=np.load(ART/'bilin18_l17_frozen_operation_allheads_rows.npz');frozen=np.load(ART/'bilin18_l17_head_specific_source_spaces_frozen_bases.npz');calids=frozen['calibration_ids'];excluded=frozen['excluded_donor_ids'];assert not set(calids)&set(excluded);model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();_,dh,_=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),cfg);means=[];center=[];meanenergy=[]
 for head in range(9):
  hh=dh[calids,head].double();mu=hh.mean(0);_,sv,vh=torch.linalg.svd(hh-mu,full_matrices=False);means.append(mu);center.append(vh[:16].T);meanenergy.append(float(mu.square().sum()/hh.square().sum(-1).mean()))
 means=torch.stack(means);center=torch.stack(center);own=torch.tensor(frozen['own'][:,:,:16],device='cuda');src=torch.tensor(old['sources'],device='cuda').double().reshape(9,12,1152);delta=src-means[:,None];fullhat=[];delhat=[]
 for head in range(9):
  u=own[head];b=center[head];h=src[head];v=delta[head];fullhat.append(torch.stack([means[head].expand_as(h),(h@u)@u.T,means[head]+(v@b)@b.T]));delhat.append(torch.stack([torch.zeros_like(v),(v@u)@u.T,(v@b)@b.T]))
 hat=torch.stack([torch.stack(fullhat,1).reshape(3,108,1152),torch.stack(delhat,1).reshape(3,108,1152)]);sources=torch.stack([src.reshape(108,1152),delta.reshape(108,1152)]);np.savez_compressed(STEM.with_name(STEM.name+'_calibration.npz'),means=means.cpu().numpy(),centered_basis=center.cpu().numpy(),calibration_ids=calids,excluded_donor_ids=excluded,mean_energy_fraction=meanenergy);print('calibration frozen',flush=True)
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();pre=torch.tensor(old['pre'],device='cuda');st=torch.tensor(old['state'],device='cuda');base=F.rms_norm(st,(1152,));ix=torch.tensor(np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz')['sets128'][0],device='cuda');xb=F.rms_norm(pre,(1152,));phib=(xb@L.T)*(xb@R.T);errors=np.empty((2,2,3,108,128,2));effects=np.empty((2,2,108,128,2));rawerr=np.empty((2,2,3,108,128));raweff=np.empty((2,2,108,128))
 for si in range(128):
  ss=st[si:si+1];th=torch.tanh(base[si:si+1].double()@W.T/30);p=(30*th).softmax(-1);cap=1-th.square();lp0=(30*th).log_softmax(-1)
  def tangent(v):
   t=(native_drms(ss.double(),v.double())@W.T)*cap;return t-(p*t).sum(-1,keepdim=True)
  def lp(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  for ai,alpha in enumerate(cfg['amplitudes']):
   for fi in range(2):
    xx=F.rms_norm(pre[si:si+1]-alpha*sources[fi].float(),(1152,));pp=(xx@L.T)*(xx@R.T);change=pp-phib[si];full=pp@D.T-phib[si]@D.T;ret=change[:,ix]@D[:,ix].T;tf=tangent(full);lt=lp(full);pt=lt.exp();effects[ai,fi,:,si]=torch.stack([.5*(p*tf.square()).sum(-1),(pt*(lt-lp0)).sum(-1)],-1).cpu().numpy();raweff[ai,fi,:,si]=full.double().square().sum(-1).cpu().numpy()
    for mi in range(3):
     xx=F.rms_norm(pre[si:si+1]-alpha*hat[fi,mi].float(),(1152,));pp=(xx@L.T)*(xx@R.T);change=pp-phib[si];om=pp@D.T-phib[si]@D.T-change[:,ix]@D[:,ix].T;v=ret+om;res=tf-tangent(v);lg=lp(v);errors[ai,fi,mi,:,si]=torch.stack([.5*(p*res.square()).sum(-1),(pt*(lt-lg)).sum(-1)],-1).cpu().numpy();rawerr[ai,fi,mi,:,si]=(full.double()-v.double()).square().sum(-1).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:,:,:,:si+1],effects=effects[:,:,:,:si+1]);print('site',si,'complete',flush=True)
 assert all(np.isfinite(v).all() for v in [errors,effects,rawerr,raweff]);prior=np.load(ART/'bilin18_l17_head_specific_source_spaces_rows.npz');replay=float(abs(errors[:,0,1]-prior['errors'][:,0]).max());effreplay=float(abs(effects[:,0]-prior['effects']).max());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,effects=effects,raw_errors=rawerr,raw_effects=raweff,source_energy=sources.square().sum(-1).reshape(2,9,12).cpu().numpy(),source_error=(hat-sources[:,None]).square().sum(-1).reshape(2,3,9,12).cpu().numpy());j={'experiment':'H461','config':cfg,'families':['full_donor','centered_deviation'],'methods':['mean_or_zero','uncentered16','affine_or_centered16'],'mean_energy_fraction':meanenergy,'reference_replay_max_abs':replay,'effect_replay_max_abs':effreplay,'numerical_gates':{'reference_replay':replay<1e-6,'effect_replay':effreplay<1e-6},'unique_completed_documents':12992};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()
