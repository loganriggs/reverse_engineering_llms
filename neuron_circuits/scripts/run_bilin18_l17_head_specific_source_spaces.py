import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_head_specific_source_spaces'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 old=np.load(ART/'bilin18_l17_frozen_operation_allheads_rows.npz');meta=json.loads((ART/'bilin18_l17_frozen_operation_allheads.json').read_text());excluded=sorted(set(meta['donor_ids']));calids=np.setdiff1d(np.arange(512),excluded);assert not set(calids)&set(excluded);model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();_,dh,_=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),cfg);own=[];sg=torch.zeros((1152,1152),device='cuda',dtype=torch.float64);spectra=[]
 for head in range(9):
  hh=dh[calids,head].double();_,sv,vh=torch.linalg.svd(hh,full_matrices=False);own.append(vh[:64].T);spectra.append(sv.cpu().numpy());g=hh.T@hh;sg+=g/g.trace()/9
 _,sh=torch.linalg.eigh(sg);shared=sh.flip(1)[:,:144];src=torch.tensor(old['sources'],device='cuda').double();own=torch.stack(own);hat=[];labels=['own16','own32','own64','shared16','shared144','frozen_head6']
 for rank in [16,32,64]:
  hh=[]
  for head in range(9):
   b=own[head,:,:rank];v=src[head*12:(head+1)*12];hh.append((v@b)@b.T)
  hat.append(torch.cat(hh))
 for rank in [16,144]:b=shared[:,:rank];hat.append((src@b)@b.T)
 Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q.double();mm=np.load(ART/'bilin18_l17_unmatched_moment_controls_moments.npz');mu=torch.tensor(mm['mean'][1],device='cuda');cov=torch.tensor(mm['covariance'][1],device='cuda');ev,U=torch.linalg.eigh(cov+mu[:,None]*mu[None,:]);inv=(U*ev.rsqrt())@U.T;V=torch.tensor(np.load(ART/'bilin18_l17_omitted_operator_shared_frozen.npz')['basis'][:,:16],device='cuda');H=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['source_directions'],device='cuda');hat.append((src@Q@inv@V)@H.T);hat=torch.stack(hat);np.savez_compressed(STEM.with_name(STEM.name+'_frozen_bases.npz'),own=own.cpu().numpy(),shared=shared.cpu().numpy(),calibration_ids=calids,excluded_donor_ids=excluded,singular_values=np.array(spectra));print('bases frozen, calibration',len(calids),flush=True)
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();pre=torch.tensor(old['pre'],device='cuda');st=torch.tensor(old['state'],device='cuda');base=F.rms_norm(st,(1152,));ix=torch.tensor(np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz')['sets128'][0],device='cuda');xb=F.rms_norm(pre,(1152,));phib=(xb@L.T)*(xb@R.T);errors=np.empty((2,6,108,128,2));effects=np.empty((2,108,128,2));rawerr=np.empty((2,6,108,128));raweff=np.empty((2,108,128))
 for si in range(128):
  ss=st[si:si+1];th=torch.tanh(base[si:si+1].double()@W.T/30);p=(30*th).softmax(-1);cap=1-th.square();lp0=(30*th).log_softmax(-1)
  def tangent(v):
   t=(native_drms(ss.double(),v.double())@W.T)*cap;return t-(p*t).sum(-1,keepdim=True)
  def lp(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  for ai,alpha in enumerate(cfg['amplitudes']):
   xx=F.rms_norm(pre[si:si+1]-alpha*src.float(),(1152,));pp=(xx@L.T)*(xx@R.T);delta=pp-phib[si];full=pp@D.T-phib[si]@D.T;ret=delta[:,ix]@D[:,ix].T;tf=tangent(full);lt=lp(full);pt=lt.exp();effects[ai,:,si]=torch.stack([.5*(p*tf.square()).sum(-1),(pt*(lt-lp0)).sum(-1)],-1).cpu().numpy();raweff[ai,:,si]=full.double().square().sum(-1).cpu().numpy()
   for mi in range(6):
    xx=F.rms_norm(pre[si:si+1]-alpha*hat[mi].float(),(1152,));pp=(xx@L.T)*(xx@R.T);delta=pp-phib[si];om=pp@D.T-phib[si]@D.T-delta[:,ix]@D[:,ix].T;v=ret+om;res=tf-tangent(v);lg=lp(v);errors[ai,mi,:,si]=torch.stack([.5*(p*res.square()).sum(-1),(pt*(lt-lg)).sum(-1)],-1).cpu().numpy();rawerr[ai,mi,:,si]=(full.double()-v.double()).square().sum(-1).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:,:,:si+1],effects=effects[:,:,:si+1]);print('site',si,'complete',flush=True)
 assert all(np.isfinite(v).all() for v in [errors,effects,rawerr,raweff]);replay=float(abs(errors[:,5]-old['errors'][:,0]).max());effect_replay=float(abs(effects-old['effects']).max());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,effects=effects,raw_errors=rawerr,raw_effects=raweff,source_error=(hat-src[None]).square().sum(-1).reshape(6,9,12).cpu().numpy(),source_energy=src.square().sum(-1).reshape(9,12).cpu().numpy());j={'experiment':'H460','config':cfg,'methods':labels,'calibration_count':len(calids),'excluded_donor_ids':excluded,'native_reference_replay_max_abs':replay,'effect_replay_max_abs':effect_replay,'numerical_gates':{'reference_replay':replay<1e-6,'effect_replay':effect_replay<1e-6},'unique_completed_documents':12992};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()
