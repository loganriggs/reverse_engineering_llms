"""H409 diagnose H408 frozen subsets without reselection."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_native_sensitivity_factors import drms
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_fisher_subset_diagnosis'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for k,v in cfg['corpus_sha256'].items():assert hashlib.sha256((ROOT/cfg[k]).read_bytes()).hexdigest()==v
 op=ART/'bilin18_l17_fisher_ranked_factors_frozen_orders.npz';fp=ART/'bilin18_l17_fisher_ranked_factors_rows.npz';assert hashlib.sha256(op.read_bytes()).hexdigest()==cfg['orders_sha256'];assert hashlib.sha256(fp.read_bytes()).hexdigest()==cfg['finite_rows_sha256'];orders=np.load(op)['orders'];ids=[torch.tensor(order[:rank],device='cuda') for order in orders for rank in cfg['ranks']];model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];Dd=D.double();W=model.lm_head.weight.double();rows=[];fullrows=[];closure=0.
 for stage in ['discovery','evaluation']:
  cc=dict(cfg,corpus=cfg[stage+'_corpus']);pre,heads,_=collect_sources(model,np.load(ROOT/cc['corpus']),cc);S=len(pre);x=F.rms_norm(pre,(1152,));phi0=(x@L.T)*(x@R.T);state=pre+phi0@D.T+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,));u=drms(pre,-heads[:,6]);phi=((u@L.T)*(x@R.T)+(x@L.T)*(u@R.T)).double();full=phi@Dd.T;kept=torch.stack([phi[:,ix]@Dd[:,ix].T for ix in ids]);m=np.empty((27,4,S));e=np.empty(S)
  for off in range(0,S,16):
   sl=slice(off,off+16);s=state[sl].double();raw=base[sl].double()@W.T;th=torch.tanh(raw/30);cap=1-th.square();p=(30*th).softmax(-1);t=(native_drms(s,full[sl])@W.T)*cap;t-= (p*t).sum(-1,keepdim=True);fe=(p*t.square()).sum(-1);e[sl]=fe.cpu().numpy()
   for oi in range(27):
    k=(native_drms(s,kept[oi,sl])@W.T)*cap;k-=(p*k).sum(-1,keepdim=True);o=t-k;ke=(p*k.square()).sum(-1);oe=(p*o.square()).sum(-1);cr=2*(p*k*o).sum(-1);at=(p*k*t).sum(-1);m[oi,:,sl]=torch.stack([ke,oe,cr,at]).cpu().numpy();closure=max(closure,float(((ke+oe+cr-fe).abs()/fe).max()))
   if (off+16)%512==0:print(stage,'scored',off+16,flush=True)
  rows.append(m.reshape(9,3,4,S));fullrows.append(e);np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),metrics=np.stack(rows),full_energy=np.stack(fullrows));print(stage,'complete',flush=True)
 m=np.stack(rows);e=np.stack(fullrows);assert np.isfinite(m).all() and np.isfinite(e).all();replay=float(np.max(np.abs(e[0]-np.load(op)['discovery_fisher'])/np.load(op)['discovery_fisher']));np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),metrics=m,full_energy=e);analyze(cfg,m,e,{'max_relative_energy_closure':closure,'H408_discovery_full_energy_relative_replay':replay})
def analyze(cfg,m,e,audit):
 n=cfg['rows'];d=m.reshape(2,9,3,4,n,4).mean(-1);ed=e.reshape(2,n,4).mean(-1);old=np.load(ART/'bilin18_l17_fisher_ranked_factors_rows.npz');finite=old['metrics'][0,1,0,0].reshape(n,4).mean(-1);eff=old['native_effect'][0].reshape(n,4).mean(-1)
 def summary(di,ei):
  discovery=1-d[0,0,1,1,di].mean()/ed[0,di].mean();evaluation=1-d[1,0,1,1,ei].mean()/ed[1,ei].mean();contrast=1-d[1,1,1,1,ei].mean()/ed[1,ei].mean();fin=1-finite[ei].mean()/eff[ei].mean();return np.array([discovery,evaluation-discovery,evaluation-fin,evaluation-contrast])
 point=summary(np.arange(n),np.arange(n));rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.stack([summary(rng.integers(0,n,n),rng.integers(0,n,n)) for _ in range(cfg['bootstrap_samples'])]);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],-1);gates={'energy_identity':audit['max_relative_energy_closure']<1e-8,'H408_discovery_Fisher_replay':audit['H408_discovery_full_energy_relative_replay']<1e-8,'discovery90':bool(ci[0,0]>.9),'transfer_equivalence05':bool(ci[1,0]>-.05 and ci[1,1]<.05),'finite_equivalence05':bool(ci[2,0]>-.05 and ci[2,1]<.05),'tangent_over_contrast05':bool(ci[3,0]>.05)};j={'config':cfg,'panels':['discovery','evaluation'],'methods':['fisher','contrast','individual_energy']+[f'random{i}' for i in range(6)],'metrics':['retained_energy','omitted_energy','cross_energy','signed_attribution'],'mean_metrics':m.mean(-1).tolist(),'mean_full_energy':e.mean(-1).tolist(),'primary_names':['fisher64_discovery_tangent_recovery','fisher64_evaluation_minus_discovery_tangent_recovery','fisher64_evaluation_tangent_minus_finite_unit_recovery','fisher64_minus_contrast64_evaluation_tangent_recovery'],'primary':point.tolist(),'primary_joint4_ci95':ci.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),primary=boot);print('FINISHED',gates,point,flush=True)
if __name__=='__main__':main()
