import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_covariance_source_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 priorids=json.loads((ART/'bilin18_l17_frozen_operation_allheads.json').read_text())['donor_ids'];pool=np.setdiff1d(np.arange(512),priorids);donors=np.random.default_rng(cfg['donor_seed']).permutation(pool)[:64];calids=np.setdiff1d(pool,donors)
 excluded=[]
 for n in ['bilin18_l17_compact_operation_confirmation_selection.json','bilin18_l17_unprojected_finite_confirmation_selection.json']:excluded+=json.loads((ART/n).read_text())['sites']
 sites=np.random.default_rng(cfg['context_seed']).permutation(np.setdiff1d(np.arange(512),excluded))[:128];assert len(sites)==128 and not set(sites)&set(excluded) and not set(calids)&set(donors)
 STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'sites':sites.tolist(),'donor_ids':donors.tolist(),'calibration_ids':calids.tolist(),'excluded_sites':excluded,'excluded_prior_donors':priorids},indent=2)+'\n')
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();_,dh,_=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),cfg)
 hh=dh[calids,6].double();mu=hh.mean(0);center=hh-mu;_,sv,vh=torch.linalg.svd(center,full_matrices=False);b=vh[:16].T;natural=dh[donors,6].double()-mu
 # Gaussian draws with exactly the empirical centered calibration covariance in expectation.
 rng=np.random.default_rng(cfg['control_seed']);g=torch.tensor(rng.standard_normal((64,len(calids))),device='cuda')@center/(len(calids)-1)**.5
 wo=model.blocks[17].attn.c_proj.weight.detach().reshape(1152,9,128)[:,6].double();iso=torch.tensor(rng.standard_normal((64,128)),device='cuda')@wo.T
 before=torch.stack([natural,g,iso]);sources=torch.stack([natural,g/g.norm(dim=-1,keepdim=True)*natural.norm(dim=-1,keepdim=True),iso/iso.norm(dim=-1,keepdim=True)*natural.norm(dim=-1,keepdim=True)])
 hat=torch.stack([(sources@b)@b.T,torch.zeros_like(sources)],1);norm_error=float((sources.norm(dim=-1)-natural.norm(dim=-1)).abs().max())
 np.savez_compressed(STEM.with_name(STEM.name+'_source_audit.npz'),basis=b.cpu().numpy(),mean=mu.cpu().numpy(),calibration_center=center.cpu().numpy(),singular_values=sv.cpu().numpy(),sources_before_norm=before.cpu().numpy(),sources=sources.cpu().numpy(),donors=donors,calibration_ids=calids)
 print('source calibration frozen;64 donors,128 unused recipient sites',flush=True)
 allpre,_,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);pre=allpre[sites];lf,rf,df=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];xx=F.rms_norm(pre,(1152,));st=pre+((xx@lf.T)*(xx@rf.T))@df.T+model.blocks[17].mlp.Down_bias
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();base=F.rms_norm(st,(1152,));ix=torch.tensor(np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz')['sets128'][0],device='cuda');xb=F.rms_norm(pre,(1152,));phib=(xb@L.T)*(xb@R.T);errors=np.empty((2,3,2,64,128,2));effects=np.empty((2,3,64,128,2));rawerr=np.empty((2,3,2,64,128));raweff=np.empty((2,3,64,128))
 for si in range(128):
  ss=st[si:si+1];th=torch.tanh(base[si:si+1].double()@W.T/30);p=(30*th).softmax(-1);cap=1-th.square();lp0=(30*th).log_softmax(-1)
  def tangent(v):
   t=(native_drms(ss.double(),v.double())@W.T)*cap;return t-(p*t).sum(-1,keepdim=True)
  def lp(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  for ai,alpha in enumerate(cfg['amplitudes']):
   for fi in range(3):
    xx=F.rms_norm(pre[si:si+1]-alpha*sources[fi].float(),(1152,));pp=(xx@L.T)*(xx@R.T);change=pp-phib[si];full=pp@D.T-phib[si]@D.T;ret=change[:,ix]@D[:,ix].T;tf=tangent(full);lt=lp(full);pt=lt.exp();effects[ai,fi,:,si]=torch.stack([.5*(p*tf.square()).sum(-1),(pt*(lt-lp0)).sum(-1)],-1).cpu().numpy();raweff[ai,fi,:,si]=full.double().square().sum(-1).cpu().numpy()
    for mi in range(2):
     xx=F.rms_norm(pre[si:si+1]-alpha*hat[fi,mi].float(),(1152,));pp=(xx@L.T)*(xx@R.T);change=pp-phib[si];om=pp@D.T-phib[si]@D.T-change[:,ix]@D[:,ix].T;v=ret+om;res=tf-tangent(v);lg=lp(v);errors[ai,fi,mi,:,si]=torch.stack([.5*(p*res.square()).sum(-1),(pt*(lt-lg)).sum(-1)],-1).cpu().numpy();rawerr[ai,fi,mi,:,si]=(full.double()-v.double()).square().sum(-1).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:,:,:,:si+1],effects=effects[:,:,:,:si+1]);print('site',si,'complete',flush=True)
 assert all(np.isfinite(v).all() for v in [errors,effects,rawerr,raweff]);np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,effects=effects,raw_errors=rawerr,raw_effects=raweff,source_energy=sources.square().sum(-1).cpu().numpy(),source_error=(hat-sources[:,None]).square().sum(-1).cpu().numpy(),pre=pre.cpu().numpy(),state=st.cpu().numpy());j={'experiment':'H463','config':cfg,'families':['natural_centered','covariance_gaussian_normmatched','isotropic_latent_normmatched'],'methods':['data16','retained_only'],'norm_match_max_abs':norm_error,'numerical_gates':{'norm_match':norm_error<1e-10,'calibration_disjoint':not bool(set(calids)&set(donors)),'unused_contexts':not bool(set(sites)&set(excluded))},'unique_completed_documents':12992};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()
