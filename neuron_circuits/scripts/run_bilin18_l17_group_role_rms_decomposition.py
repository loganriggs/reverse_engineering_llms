"""H418 head6 input-RMS decomposition in fixed native output Fisher metric."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_group_role_rms_decomposition'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());op=ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz';mp=ART/'bilin18_l17_group_role_within_panel_donor_mappings.npz';assert hashlib.sha256(op.read_bytes()).hexdigest()==cfg['sets_sha256'];assert hashlib.sha256(mp.read_bytes()).hexdigest()==cfg['prior_maps_sha256'];assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256'];q=np.load(op);ids=[torch.tensor(v,device='cuda') for v in [q['core'],q['sets128'][0,64:]]];maps=np.load(mp)['document_permutations'];model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);S=len(pre);L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();xf=F.rms_norm(pre,(1152,));state=pre+((xf@L.T)*(xf@R.T))@D.T+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,));pd=pre.double();Ld,Rd,Dd=L.double(),R.double(),D.double();s=(pd.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pd/s;lx=x@Ld.T;rx=x@Rd.T;phi0=lx*rx;ref=heads[:,6];Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q;rows=np.empty((25,5,S));decomp=0.;closure=0.
 for condition in range(25):
  if condition==0:pay=ref
  else:
   family=(condition-1)//6;di=(condition-1)%6;donor=torch.tensor(maps[di],device='cuda');site=(donor[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);raw=heads[site,6];rn=ref.norm(dim=-1,keepdim=True);dn=raw.norm(dim=-1,keepdim=True)
   if family==0:pay=raw
   elif family==1:pay=raw*(rn/dn)
   elif family==2:pay=angle_matched_payload(pre,ref*(dn/rn),Q,raw@Q)
   else:pay=angle_matched_payload(pre,ref,Q,raw@Q)
  hd=pay.double();fixed_u=-hd/s;alpha=(pd*hd).mean(-1,keepdim=True)/s.square();pf=(fixed_u@Ld.T)*rx+lx*(fixed_u@Rd.T);pr=2*alpha*phi0;u=native_drms(pd,-hd);direct=(u@Ld.T)*rx+lx*(u@Rd.T);full=direct@Dd.T;vs=[full];norm=full.norm(dim=-1)
  for ix in ids:
   gf=pf[:,ix]@Dd[:,ix].T;gr=pr[:,ix]@Dd[:,ix].T;truth=direct[:,ix]@Dd[:,ix].T;decomp=max(decomp,float(((gf+gr-truth).norm(dim=-1)/norm).max()));vs.extend([gf,gr])
  for off in range(0,S,16):
   sl=slice(off,off+16);st=state[sl].double();th=torch.tanh((base[sl].double()@W.T)/30);cap=1-th.square();p=(30*th).softmax(-1);ts=[]
   for v in vs:
    t=(native_drms(st,v[sl])@W.T)*cap;t-=(p*t).sum(-1,keepdim=True);ts.append(t)
   total,cf,cr,af,ar=ts;energy=(p*total.square()).sum(-1);terms=torch.stack([2*(p*cf*af).sum(-1),2*(p*cf*ar).sum(-1),2*(p*cr*af).sum(-1),2*(p*cr*ar).sum(-1)]);cross=2*(p*(cf+cr)*(af+ar)).sum(-1);closure=max(closure,float(((terms.sum(0)-cross).abs()/energy).max()));rows[condition,:,sl]=torch.cat([energy[None],terms],0).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),metrics=rows[:condition+1]);print('condition',condition,'complete',flush=True)
 assert np.isfinite(rows).all();audit={'group_derivative_decomposition_max_relative':decomp,'cross_term_closure_full_energy_normalized':closure};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),metrics=rows);STEM.with_name(STEM.name+'_audit.json').write_text(json.dumps(audit,indent=2)+'\n');analyze(cfg,rows,audit)
def analyze(cfg,rows,audit):
 fp=ART/'bilin18_l17_group_role_norm_angle_factorial.json';assert hashlib.sha256(fp.read_bytes()).hexdigest()==cfg['prior_factorial_sha256'];prior=json.loads(fp.read_text());old=json.loads((ART/'bilin18_l17_group_role_within_panel.json').read_text());n=cfg['rows'];d=rows.reshape(25,5,n,4).mean(-1)
 def summary(docs,draw):
  nums=[]
  for fi in range(4):
   v=d[1+6*fi+draw][:,:,docs].mean((0,2));nums.append(v[1:]/v[0])
  nums=np.stack(nums);delta=np.stack([nums[2]-nums[0],nums[3]-nums[1]]);tot=delta.sum(-1);radial=delta[:,1:].sum(-1)/tot;return np.r_[tot,radial],nums,delta
 point,nums,delta=summary(np.arange(n),np.arange(6));actual=rows[0,1:].mean(-1)/rows[0,0].mean();replay=max(abs(float(actual.sum())-old['actual_roles'][0]),float(np.max(abs(nums.sum(-1)-np.array(prior['roles'])[:,0]))));audit['pooled_role_float_replay_max_absolute']=replay;rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.stack([summary(rng.integers(0,n,n),rng.integers(0,6,6))[0] for _ in range(cfg['bootstrap_samples'])]);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],-1);gates={'decomposition_identity':audit['group_derivative_decomposition_max_relative']<1e-10 and audit['cross_term_closure_full_energy_normalized']<1e-10,'pooled_replay':replay<1e-5,'angle_effect10':bool((ci[:2,0]>.1).all()),'radial_accounts80':bool((ci[2:,0]>.8).all()),'fixed_accounts80':bool((ci[2:,1]<.2).all())};j={'config':cfg,'conditions':['actual','raw','norm_only','angle_only','both'],'terms':['fixed_fixed','fixed_radial','radial_fixed','radial_radial'],'normalized_cross_terms':np.vstack([actual,nums]).tolist(),'angle_contributions_donor_and_recipient_norms':delta.tolist(),'primary_names':['angle_effect_donor_norm','angle_effect_recipient_norm','radial_fraction_donor_norm','radial_fraction_recipient_norm'],'primary':point.tolist(),'primary_joint4_ci95':ci.tolist(),'gates':gates,'audit':audit};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),primary=boot);print('FINISHED',gates,point,flush=True)
if __name__=='__main__':main()
