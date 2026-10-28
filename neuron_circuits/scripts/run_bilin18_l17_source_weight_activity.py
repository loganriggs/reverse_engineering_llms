import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_source_weight_activity'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 old=np.load(ART/'bilin18_l17_frozen_operation_allheads_rows.npz');cal=np.load(ART/'bilin18_l17_source_mean_variation_calibration.npz');model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();means=torch.tensor(cal['means'],device='cuda');data=torch.tensor(cal['centered_basis'],device='cuda');src=torch.tensor(old['sources'],device='cuda').double().reshape(9,12,1152);natural=src-means[:,None];weight=[];isotropic=[];spectra=[];data_capture=[];weight_capture=[]
 for head in range(9):
  wo=model.blocks[17].attn.c_proj.weight.detach().reshape(1152,9,128)[:,head].double();u,sv,_=torch.linalg.svd(wo,full_matrices=False);weight.append(u[:,:16]);spectra.append(sv.cpu().numpy());data_capture.append(float((data[head].T@wo).square().sum()/wo.square().sum()));weight_capture.append(float(sv[:16].square().sum()/sv.square().sum()));noise=torch.tensor(np.random.default_rng(cfg['control_seed']+head).standard_normal((12,128)),device='cuda');v=noise@wo.T;v=v/v.norm(dim=-1,keepdim=True)*natural[head].norm(dim=-1,keepdim=True);isotropic.append(v)
 weight=torch.stack(weight);isotropic=torch.stack(isotropic);sources=torch.stack([natural.reshape(108,1152),isotropic.reshape(108,1152)]);hat=[]
 for family in [natural,isotropic]:
  datahat=[];weighthat=[]
  for head in range(9):
   v=family[head];b=data[head];w=weight[head];datahat.append((v@b)@b.T);weighthat.append((v@w)@w.T)
  hat.append(torch.stack([torch.cat(datahat),torch.cat(weighthat),torch.zeros_like(sources[0])]))
 hat=torch.stack(hat);norm_error=float((sources[0].norm(dim=-1)-sources[1].norm(dim=-1)).abs().max());np.savez_compressed(STEM.with_name(STEM.name+'_source_audit.npz'),weight_basis=weight.cpu().numpy(),singular_values=np.array(spectra),data_expected_isotropic_capture=data_capture,weight_expected_isotropic_capture=weight_capture,sources=sources.cpu().numpy());print('weight/source audit frozen',flush=True)
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
 assert all(np.isfinite(v).all() for v in [errors,effects,rawerr,raweff]);prior=np.load(ART/'bilin18_l17_source_mean_variation_rows.npz');replay=float(abs(errors[:,0,0]-prior['errors'][:,1,2]).max());effreplay=float(abs(effects[:,0]-prior['effects'][:,1]).max());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,effects=effects,raw_errors=rawerr,raw_effects=raweff,source_energy=sources.square().sum(-1).reshape(2,9,12).cpu().numpy(),source_error=(hat-sources[:,None]).square().sum(-1).reshape(2,3,9,12).cpu().numpy());j={'experiment':'H462','config':cfg,'families':['natural_centered','isotropic_latent_normmatched'],'methods':['data16','weight16','retained_only'],'weight_spectral_capture16':weight_capture,'data_expected_isotropic_capture16':data_capture,'norm_match_max_abs':norm_error,'reference_replay_max_abs':replay,'effect_replay_max_abs':effreplay,'numerical_gates':{'norm_match':norm_error<1e-10,'reference_replay':replay<1e-6,'effect_replay':effreplay<1e-6},'unique_completed_documents':12992};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()
