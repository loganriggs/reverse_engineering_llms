"""H420 fresh directional recombination and finite group prediction."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_component_recombination'
def pair_maps(n,seed):
 rng=np.random.default_rng(seed);base=np.arange(n)
 while True:
  a=rng.permutation(n)
  if np.all(a!=base):break
 while True:
  b=rng.permutation(n)
  if np.all(b!=base) and np.all(b!=a):return a,b
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());op=ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz';assert hashlib.sha256(op.read_bytes()).hexdigest()==cfg['sets_sha256'];assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256'];q=np.load(op);ids=[torch.tensor(v,device='cuda') for v in [q['core'],q['sets128'][0,64:],q['sets128'][0]]];maps=np.stack([pair_maps(cfg['rows'],seed) for seed in cfg['draw_seeds']]);np.savez_compressed(STEM.with_name(STEM.name+'_donor_maps.npz'),maps=maps);model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);S=len(pre);L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();xf=F.rms_norm(pre,(1152,));phi0=(xf@L.T)*(xf@R.T);m0=phi0@D.T;state=pre+m0+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,));pd=pre.double();Ld,Rd,Dd=L.double(),R.double(),D.double();scale=(pd.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pd/scale;lx=x@Ld.T;rx=x@Rd.T;Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q;axis=(pd@Q.double())@Q.double().T;axis/=axis.norm(dim=-1,keepdim=True);tangent=np.empty((19,3,S));finite=np.empty((19,3,2,S));effect=np.empty((19,S));audit={'norm_error':0.,'dot_error':0.,'span_error':0.,'self_vector_error':0.,'clamp_state_error':0.}
 def lp(v):return (30*torch.tanh((v.double()@W.T)/30)).log_softmax(-1)
 for condition in range(19):
  if condition==0:pay=heads[:,6]
  else:
   family=(condition-1)//6;di=(condition-1)%6;sa=(torch.tensor(maps[di,0],device='cuda')[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);sb=(torch.tensor(maps[di,1],device='cuda')[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);raw=heads[sa,6]
   if family==0:pay=raw
   else:
    donor=raw if family==1 else heads[sb,6];pay=angle_matched_payload(pre,raw,Q,donor@Q);rn=raw.double().norm(dim=-1);audit['norm_error']=max(audit['norm_error'],float(((pay.double().norm(dim=-1)-rn).abs()/rn).max()));audit['dot_error']=max(audit['dot_error'],float(((pd*(pay-raw).double()).sum(-1).abs()/(pd.norm(dim=-1)*rn)).max()));audit['span_error']=max(audit['span_error'],float(((pay-(pay@Q)@Q.T).double().norm(dim=-1)/rn).max()))
    if family==1:audit['self_vector_error']=max(audit['self_vector_error'],float(((pay-raw).double().norm(dim=-1)/rn).max()))
  hd=pay.double();u=native_drms(pd,-hd);phi=(u@Ld.T)*rx+lx*(u@Rd.T);vs=[phi@Dd.T]+[phi[:,ix]@Dd[:,ix].T for ix in ids[:2]];par=(hd*axis).sum(-1,keepdim=True)*axis;perp=hd-par;parts=[]
  for v in [par,perp]:
   uu=-v/scale;parts.append((uu@Ld.T)*rx+lx*(uu@Rd.T))
  vs.extend([part[:,ix]@Dd[:,ix].T for ix in ids[:2] for part in parts]);xx=F.rms_norm(pre-pay,(1152,));phi1=(xx@L.T)*(xx@R.T);r=phi1@D.T-m0;delta=phi1-phi0;kept=torch.stack([delta[:,ix]@D[:,ix].T for ix in ids]);target=F.rms_norm(state+r,(1152,));pred=F.rms_norm(state[None]+kept,(1152,));clamp=F.rms_norm(state[None]+r[None]-kept,(1152,))
  for gi,ix in enumerate(ids):
   patch=phi1[:4].clone();patch[:,ix]=phi0[:4,ix];direct=F.rms_norm(pre[:4]+patch@D.T+model.blocks[17].mlp.Down_bias,(1152,));audit['clamp_state_error']=max(audit['clamp_state_error'],float((direct-clamp[gi,:4]).abs().max()))
  for off in range(0,S,16):
   sl=slice(off,off+16);st=state[sl].double();th=torch.tanh((base[sl].double()@W.T)/30);cap=1-th.square();p=(30*th).softmax(-1);ts=[]
   for v in vs:
    dt=(native_drms(st,v[sl])@W.T)*cap;dt-=(p*dt).sum(-1,keepdim=True);ts.append(dt)
   total,c,a,cp,co,ap,ao=ts;tangent[condition,:,sl]=torch.stack([(p*total.square()).sum(-1),2*(p*c*a).sum(-1),2*(p*(cp*ao+co*ap)).sum(-1)]).cpu().numpy();pt=lp(target[sl]);p0=(30*th).log_softmax(-1);effect[condition,sl]=(pt.exp()*(pt-p0)).sum(-1).cpu().numpy()
   for gi in range(3):
    pp=lp(pred[gi,sl]);pc=lp(clamp[gi,sl]);finite[condition,gi,0,sl]=(pt.exp()*(pt-pp)).sum(-1).cpu().numpy();finite[condition,gi,1,sl]=(pc.exp()*(pc-p0)).sum(-1).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),tangent=tangent[:condition+1],finite=finite[:condition+1],effect=effect[:condition+1]);print('condition',condition,'complete',flush=True)
 assert np.isfinite(tangent).all() and np.isfinite(finite).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),tangent=tangent,finite=finite,effect=effect);STEM.with_name(STEM.name+'_audit.json').write_text(json.dumps(audit,indent=2)+'\n');analyze(cfg,tangent,finite,effect,audit)
def analyze(cfg,t,k,e,audit):
 n=cfg['rows'];td=t.reshape(19,3,n,4).mean(-1);kd=k.reshape(19,3,2,n,4).mean(-1);ed=e.reshape(19,n,4).mean(-1)
 def aggregate(ix,docs):
  v=td[ix][:,:,docs].mean((0,2));kk=kd[ix][:,:,:,docs].mean((0,3));ee=ed[ix][:,docs].mean();rec=1-kk[:,0]/ee;benefit=rec[2]-rec[0]-rec[1];return v,rec,kk[:,1]/ee,benefit
 def summary(docs,draw):
  raw=aggregate(1+draw,docs);cross=aggregate(13+draw,docs);return np.array([cross[0][1]/cross[0][0]-raw[0][1]/raw[0][0],(cross[0][2]-raw[0][2])/raw[0][0],raw[3]-cross[3]])
 point=summary(np.arange(n),np.arange(6));rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.stack([summary(rng.integers(0,n,n),rng.integers(0,6,6)) for _ in range(cfg['bootstrap_samples'])]);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],-1);families=[aggregate(ix,np.arange(n)) for ix in [[0],np.arange(1,7),np.arange(7,13),np.arange(13,19)]];raw,self=families[1:3];audit['self_pooled_score_error']=float(max(abs(raw[0][1]/raw[0][0]-self[0][1]/self[0][0]),abs(raw[1]-self[1]).max(),abs(raw[2]-self[2]).max()));doc=json.loads((ART/'bilin18_component_recombination_document_audit.json').read_text());gates={'document_disjointness':doc['overlap']==0 and doc['independently_rehashed_excluded_documents']==8896,'geometry':max(audit[k] for k in ['norm_error','dot_error','span_error'])<1e-5,'self_reconstruction':audit['self_vector_error']<1e-5 and audit['self_pooled_score_error']<1e-4,'cross_shift10':bool(ci[0,0]>.1),'mixed_shift10':bool(ci[1,0]>.1),'finite_combination_loss10':bool(ci[2,0]>.1)};j={'config':cfg,'families':['actual','raw','self_reconstructed','cross_recombined'],'groups':['core64','added64','union128'],'family_tangent_means':[v[0].tolist() for v in families],'family_recoveries':[v[1].tolist() for v in families],'family_clamp_remaining':[v[2].tolist() for v in families],'family_combination_benefit':[float(v[3]) for v in families],'primary_names':['cross_minus_raw_full_cross_ratio','mixed_numerator_shift_over_raw_full_energy','raw_minus_cross_finite_combination_benefit'],'primary':point.tolist(),'primary_joint3_ci95':ci.tolist(),'gates':gates,'audit':audit,'document_audit':doc};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),primary=boot);print('FINISHED',gates,point,flush=True)
if __name__=='__main__':main()
