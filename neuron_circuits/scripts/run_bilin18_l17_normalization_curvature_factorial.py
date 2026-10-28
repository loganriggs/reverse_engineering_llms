"""H426 input RMS versus bilinear curvature."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_normalization_curvature_factorial'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());op=ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz';assert hashlib.sha256(op.read_bytes()).hexdigest()==cfg['sets_sha256'];assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256'];q=np.load(op);ids=[torch.tensor(v,device='cuda') for v in [q['core'],q['sets128'][0,64:],q['sets128'][0]]];maps=np.stack([np.concatenate([np.random.default_rng(seed+10000*block).permutation(cfg['donor_rows']) for block in range(2)]) for seed in cfg['draw_seeds']]);np.savez_compressed(STEM.with_name(STEM.name+'_donor_maps.npz'),maps=maps);model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);assert hashlib.sha256((ROOT/cfg['donor_corpus']).read_bytes()).hexdigest()==cfg['donor_sha256'];_,donor_heads,_=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),dict(cfg,rows=cfg['donor_rows']));S=len(pre);L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();xf=F.rms_norm(pre,(1152,));phi0=(xf@L.T)*(xf@R.T);m0=phi0@D.T;state=pre+m0+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,));pd=pre.double();Ld,Rd,Dd=L.double(),R.double(),D.double();scale=(pd.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pd/scale;lx=x@Ld.T;rx=x@Rd.T;Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q;axis=(pd@Q.double())@Q.double().T;axis/=axis.norm(dim=-1,keepdim=True);tangent=np.empty((13,3,S));finite=np.empty((13,3,2,S));effect=np.empty((13,S));audit={'clamp_state_error':0.}
 def lp(v):return (30*torch.tanh((v.double()@W.T)/30)).log_softmax(-1)
 oldstem=ART/'bilin18_l17_unmatched_moment_controls';mp=oldstem.with_name(oldstem.name+'_moments.npz');dm=oldstem.with_name(oldstem.name+'_donor_maps.npz');assert hashlib.sha256(mp.read_bytes()).hexdigest()==cfg['moments_sha256'];m=np.load(mp);mu=torch.tensor(m['mean'],device='cuda');root=torch.tensor(m['root'],device='cuda');amps=cfg['amplitudes'];finite=np.empty((len(amps),13,3,S));effect=np.empty((len(amps),13,S));audit={'group_partition_relative_error':0.}
 reference=ART/'bilin18_l17_response_metric_confirmation_rows.npz';assert hashlib.sha256(reference.read_bytes()).hexdigest()==cfg['reference_rows_sha256'];ref=np.load(reference);errors=np.empty((2,2,2,25,3,S));energies=np.empty((2,2,2,25,S));errors[1,1]=ref['errors'][1,1];energies[1,1]=ref['energies'][1,1];audit={'bilinear_identity':0.,'linear_baseline_vector':0.}
 for condition in range(25):
  if condition==0:pay=heads[:,6]
  else:
   family=(condition-1)//12;di=(condition-1)%12
   if family==0:
    sa=(torch.tensor(maps[di],device='cuda')[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=donor_heads[sa,6]
   else:
    noise=torch.tensor(np.random.default_rng(cfg['draw_seeds'][di]+100000).standard_normal((cfg['rows'],4,128)),device='cuda');cc=mu+torch.einsum('npi,pji->npj',noise,root);pay=(cc.reshape(S,128)@Q.double().T).float()
  hd=pay.double();du=native_drms(pd,-hd);dphi=(du@Ld.T)*rx+lx*(du@Rd.T)
  for ai,amp in enumerate(amps):
   pp=pd-amp*hd;un=pp/(pp.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()-x
   for ni,u in enumerate([un,amp*du]):
    lu=u@Ld.T;ru=u@Rd.T;first=lu*rx+lx*ru;second=lu*ru
    if ni==1:audit['linear_baseline_vector']=max(audit['linear_baseline_vector'],float(((first-amp*dphi).norm(dim=-1)/(amp*dphi.norm(dim=-1)+1e-12)).max()))
    direct=(lx+lu)*(rx+ru)-lx*rx;audit['bilinear_identity']=max(audit['bilinear_identity'],float(((direct-first-second).norm(dim=-1)/((first+second).norm(dim=-1)+1e-12)).max()))
    for bi,delta in enumerate([first+second,first]):
     if ni==1 and bi==1:continue
     full=delta@Dd.T;part=torch.stack([delta[:,ix]@Dd[:,ix].T for ix in ids])
     for off in range(0,S,16):
      sl=slice(off,off+16);st=state[sl].double();th=torch.tanh((base[sl].double()@W.T)/30);cap=1-th.square();p=(30*th).softmax(-1)
      def tangent(v):
       dt=(native_drms(st,v)@W.T)*cap;return dt-(p*dt).sum(-1,keepdim=True)
      total=tangent(full[sl]);energies[ni,bi,ai,condition,sl]=(.5*(p*total.square()).sum(-1)).cpu().numpy()
      for gi in range(3):
       residual=total-tangent(part[gi,sl]);errors[ni,bi,ai,condition,gi,sl]=(.5*(p*residual.square()).sum(-1)).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:,:,:condition+1],energies=energies[:,:,:,:condition+1]);STEM.with_name(STEM.name+'_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print('condition',condition,'complete',flush=True)
 assert np.isfinite(errors).all() and np.isfinite(energies).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,energies=energies);analyze(cfg,errors,energies,audit)
def analyze(cfg,k,e,audit):
 n=cfg['rows'];kd=k.reshape(2,2,2,25,3,n,4).mean(-1);ed=e.reshape(2,2,2,25,n,4).mean(-1)
 def agg(ix,docs):
  kk=kd[:,:,:,ix][:,:,:,:,:,docs].mean((3,5));ee=ed[:,:,:,ix][:,:,:,:,docs].mean((3,4));rec=1-kk/ee[:,:,:,None];return rec,ee,rec[:,:,:,2]-rec[:,:,:,0]-rec[:,:,:,1]
 def all_interactions(docs,draw):
  raw=agg(1+draw,docs);gauss=agg(13+draw,docs);diff=gauss[2]-raw[2];return diff[:,:,1]-diff[:,:,0]
 def contrast(docs,draw):
  ii=all_interactions(docs,draw);return np.array([ii[0,1]-ii[1,1],ii[1,0]-ii[1,1],ii[0,0]-ii[0,1]-ii[1,0]+ii[1,1]])
 docs=np.arange(n);draw=np.arange(12);point=contrast(docs,draw);ii=all_interactions(docs,draw);rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.stack([contrast(rng.integers(0,n,n),rng.integers(0,12,12)) for _ in range(cfg['bootstrap_samples'])]);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],-1);fam=[agg(ix,docs) for ix in [[0],np.arange(1,13),np.arange(13,25)]];old=json.loads((ART/'bilin18_l17_response_metric_confirmation.json').read_text());oldS=np.array(old['family_combination_scores'])[:,0,1];oldE=np.array(old['family_energies'])[:,0,1];newS=np.array([v[2][0,0] for v in fam]);newE=np.array([v[1][0,0] for v in fam]);audit['native_pooled_score_replay']=float(abs(newS-oldS).max());audit['native_pooled_energy_relative_replay']=float(abs(newE/oldE-1).max());audit['factorial_closure']=float(abs(point.sum()-(ii[0,0]-ii[1,1])));gates={'bilinear_identity':audit['bilinear_identity']<1e-9,'linear_baseline_vector':audit['linear_baseline_vector']<1e-10,'native_pooled_replay':max(audit['native_pooled_score_replay'],audit['native_pooled_energy_relative_replay'])<1e-5};j={'config':cfg,'normalization_order':['exact','linear'],'bilinear_order':['full','linear'],'families':['actual','raw','gaussian'],'family_recoveries':[v[0].tolist() for v in fam],'family_energies':[v[1].tolist() for v in fam],'family_combination_scores':[v[2].tolist() for v in fam],'distribution_amplitude_interactions':ii.tolist(),'primary_names':['input_RMS_only','bilinear_only','factorial_interaction'],'primary':point.tolist(),'primary_joint3_ci95':ci.tolist(),'negative02':(ci[:,1]<-.02).tolist(),'equivalence02':(abs(ci)<.02).all(-1).tolist(),'audit':audit,'numerical_gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),primary=boot);print('FINISHED',point,ci,audit,flush=True)
if __name__=='__main__':main()
