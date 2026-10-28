import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_source_norm_policy'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 z=np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz');old=np.load(ART/'bilin18_l17_covariance_source_confirmation_rows.npz');natural=torch.tensor(z['sources'][0],device='cuda');g=torch.tensor(z['sources_before_norm'][1],device='cuda');paired=torch.tensor(z['sources'][1],device='cuda');b=torch.tensor(z['basis'],device='cuda');nn=natural.norm(dim=-1);gn=g.norm(dim=-1);assigned=torch.empty_like(nn);assigned[torch.argsort(gn)]=torch.sort(nn).values;sortedg=g/gn[:,None]*assigned[:,None];globalg=g*(nn.square().sum()/gn.square().sum()).sqrt();sources=torch.stack([natural,paired,sortedg,globalg]);hat=torch.stack([(sources@b)@b.T,torch.zeros_like(sources)],1)
 energy_error=float((sources.square().sum((1,2))/natural.square().sum()-1).abs().max());sorted_error=float((torch.sort(sortedg.norm(dim=-1)).values-torch.sort(nn).values).abs().max());direction_error=float((sortedg/sortedg.norm(dim=-1,keepdim=True)-g/gn[:,None]).abs().max());assert energy_error<1e-12 and sorted_error<1e-10 and direction_error<1e-12
 np.savez_compressed(STEM.with_name(STEM.name+'_source_audit.npz'),sources=sources.cpu().numpy(),basis=b.cpu().numpy(),sorted_assigned_norms=assigned.cpu().numpy())
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre=torch.tensor(old['pre'],device='cuda');st=torch.tensor(old['state'],device='cuda');print('normalization policies frozen',flush=True)
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();base=F.rms_norm(st,(1152,));ix=torch.tensor(np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz')['sets128'][0],device='cuda');xb=F.rms_norm(pre,(1152,));phib=(xb@L.T)*(xb@R.T);errors=np.empty((2,4,2,64,128,2));effects=np.empty((2,4,64,128,2));rawerr=np.empty((2,4,2,64,128));raweff=np.empty((2,4,64,128))
 for si in range(128):
  ss=st[si:si+1];th=torch.tanh(base[si:si+1].double()@W.T/30);p=(30*th).softmax(-1);cap=1-th.square();lp0=(30*th).log_softmax(-1)
  def tangent(v):
   t=(native_drms(ss.double(),v.double())@W.T)*cap;return t-(p*t).sum(-1,keepdim=True)
  def lp(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  for ai,alpha in enumerate(cfg['amplitudes']):
   for fi in range(4):
    xx=F.rms_norm(pre[si:si+1]-alpha*sources[fi].float(),(1152,));pp=(xx@L.T)*(xx@R.T);change=pp-phib[si];full=pp@D.T-phib[si]@D.T;ret=change[:,ix]@D[:,ix].T;tf=tangent(full);lt=lp(full);pt=lt.exp();effects[ai,fi,:,si]=torch.stack([.5*(p*tf.square()).sum(-1),(pt*(lt-lp0)).sum(-1)],-1).cpu().numpy();raweff[ai,fi,:,si]=full.double().square().sum(-1).cpu().numpy()
    for mi in range(2):
     xx=F.rms_norm(pre[si:si+1]-alpha*hat[fi,mi].float(),(1152,));pp=(xx@L.T)*(xx@R.T);change=pp-phib[si];om=pp@D.T-phib[si]@D.T-change[:,ix]@D[:,ix].T;v=ret+om;res=tf-tangent(v);lg=lp(v);errors[ai,fi,mi,:,si]=torch.stack([.5*(p*res.square()).sum(-1),(pt*(lt-lg)).sum(-1)],-1).cpu().numpy();rawerr[ai,fi,mi,:,si]=(full.double()-v.double()).square().sum(-1).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:,:,:,:si+1],effects=effects[:,:,:,:si+1]);print('site',si,'complete',flush=True)
 assert all(np.isfinite(v).all() for v in [errors,effects,rawerr,raweff]);np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,effects=effects,raw_errors=rawerr,raw_effects=raweff,source_energy=sources.square().sum(-1).cpu().numpy(),source_error=(hat-sources[:,None]).square().sum(-1).cpu().numpy(),pre=pre.cpu().numpy(),state=st.cpu().numpy());replay=float(abs(errors[:,:2]-old['errors'][:,:2]).max());effreplay=float(abs(effects[:,:2]-old['effects'][:,:2]).max());j={'experiment':'H464','config':cfg,'families':['natural','paired_norm','sorted_norm','global_scale'],'methods':['data16','retained_only'],'energy_relative_error':energy_error,'sorted_norm_max_abs':sorted_error,'direction_max_abs':direction_error,'replay_max_abs':replay,'effect_replay_max_abs':effreplay,'numerical_gates':{'energy':energy_error<1e-12,'sorted_norm':sorted_error<1e-10,'direction':direction_error<1e-12,'replay':replay<1e-6,'effect_replay':effreplay<1e-6},'unique_completed_documents':12992};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()
