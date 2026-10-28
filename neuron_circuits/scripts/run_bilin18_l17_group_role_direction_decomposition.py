"""H419 fixed-scale parallel/orthogonal source decomposition."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_group_role_direction_decomposition'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());op=ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz';mp=ART/'bilin18_l17_group_role_within_panel_donor_mappings.npz';rp=ART/'bilin18_l17_group_role_rms_decomposition_rows.npz'
 for p,h in [(op,cfg['sets_sha256']),(mp,cfg['prior_maps_sha256']),(rp,cfg['rms_rows_sha256']),(ROOT/cfg['corpus'],cfg['corpus_sha256'])]:assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 prior=np.load(rp)['metrics'];q=np.load(op);ids=[torch.tensor(v,device='cuda') for v in [q['core'],q['sets128'][0,64:]]];maps=np.load(mp)['document_permutations'];model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);S=len(pre);L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();xf=F.rms_norm(pre,(1152,));state=pre+((xf@L.T)*(xf@R.T))@D.T+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,));pd=pre.double();Ld,Rd,Dd=L.double(),R.double(),D.double();s=(pd.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pd/s;lx=x@Ld.T;rx=x@Rd.T;ref=heads[:,6];Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q;axis=(pd@Q.double())@Q.double().T;axis=axis/axis.norm(dim=-1,keepdim=True);rows=np.empty((25,3,S));coeff=np.empty((25,S));orth=0.
 for condition in range(25):
  if condition==0:pay=ref
  else:
   family=(condition-1)//6;di=(condition-1)%6;donor=torch.tensor(maps[di],device='cuda');site=(donor[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);raw=heads[site,6];rn=ref.norm(dim=-1,keepdim=True);dn=raw.norm(dim=-1,keepdim=True)
   if family==0:pay=raw
   elif family==1:pay=raw*(rn/dn)
   elif family==2:pay=angle_matched_payload(pre,ref*(dn/rn),Q,raw@Q)
   else:pay=angle_matched_payload(pre,ref,Q,raw@Q)
  hd=pay.double();amplitude=(hd*axis).sum(-1,keepdim=True);parallel=amplitude*axis;perp=hd-parallel;coeff[condition]=amplitude.squeeze(-1).cpu().numpy();orth=max(orth,float(((perp*axis).sum(-1).abs()/hd.norm(dim=-1)).max()));phis=[]
  for part in [parallel,perp]:
   u=-part/s;phis.append((u@Ld.T)*rx+lx*(u@Rd.T))
  vs=[phi[:,ix]@Dd[:,ix].T for ix in ids for phi in phis]
  for off in range(0,S,16):
   sl=slice(off,off+16);st=state[sl].double();th=torch.tanh((base[sl].double()@W.T)/30);cap=1-th.square();p=(30*th).softmax(-1);ts=[]
   for v in vs:
    t=(native_drms(st,v[sl])@W.T)*cap;t-=(p*t).sum(-1,keepdim=True);ts.append(t)
   cp,co,ap,ao=ts;rows[condition,:,sl]=torch.stack([2*(p*cp*ap).sum(-1),2*(p*(cp*ao+co*ap)).sum(-1),2*(p*co*ao).sum(-1)]).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),components=rows[:condition+1],parallel_amplitude=coeff[:condition+1]);print('condition',condition,'complete',flush=True)
 norm=ref.double().norm(dim=-1).cpu().numpy();audit={'orthogonality_scaled_error':orth,'actual_both_parallel_amplitude_norm_scaled_error':float(np.max(abs(coeff[19:25]-coeff[0])/norm)),'FF_reconstruction_full_energy_normalized_max_error':float(np.max(abs(rows.sum(1)-prior[:,1])/prior[:,0]))};assert np.isfinite(rows).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),components=rows,parallel_amplitude=coeff,full_energy=prior[:,0]);STEM.with_name(STEM.name+'_audit.json').write_text(json.dumps(audit,indent=2)+'\n');analyze(cfg,rows,prior[:,0],audit)
def analyze(cfg,rows,energy,audit):
 n=cfg['rows'];d=rows.reshape(25,3,n,4).mean(-1);f=energy.reshape(25,n,4).mean(-1)
 def summary(docs,draw):
  components=[]
  for fi in range(4):
   ix=1+6*fi+draw;components.append(d[ix][:,:,docs].mean((0,2))/f[ix][:,docs].mean())
  comp=np.stack(components);change=np.stack([comp[2]-comp[0],comp[3]-comp[1]]);fractions=change/change.sum(-1,keepdims=True);actual=d[0][:,docs].mean(-1)/f[0,docs].mean();both=d[19+draw][:,:,docs].mean((0,2))/f[0,docs].mean();gap=both-actual;gapfrac=gap/gap.sum();return fractions,gapfrac,comp,change,gap
 point,gp,comp,change,gap=summary(np.arange(n),np.arange(6));rng=np.random.default_rng(cfg['bootstrap_seed']);boot=[];gb=[]
 for _ in range(cfg['bootstrap_samples']):
  b,g,*_=summary(rng.integers(0,n,n),rng.integers(0,6,6));boot.append(b);gb.append(g)
 boot=np.stack(boot);gb=np.stack(gb);rad=np.quantile(abs(boot-point).max((1,2)),.95);gr=np.quantile(abs(gb-gp).max(1),.95);ci=np.stack([point-rad,point+rad],-1);gci=np.stack([gp-gr,gp+gr],-1);gates={'replay':audit['FF_reconstruction_full_energy_normalized_max_error']<1e-9,'parallel_match':audit['actual_both_parallel_amplitude_norm_scaled_error']<1e-5,'angle_parallel80':bool((ci[:,0,0]>.8).all()),'angle_mixed80':bool((ci[:,1,0]>.8).all()),'angle_orthogonal80':bool((ci[:,2,0]>.8).all()),'actual_mixed80':bool(gci[1,0]>.8),'actual_orthogonal80':bool(gci[2,0]>.8)};actual=rows[0].mean(-1)/energy[0].mean();j={'config':cfg,'conditions':['actual','raw','norm_only','angle_only','both'],'components':['parallel_parallel','mixed','orthogonal_orthogonal'],'normalized_components':np.vstack([actual,comp]).tolist(),'angle_component_changes':change.tolist(),'angle_component_fractions':point.tolist(),'angle_fractions_joint6_ci95':ci.tolist(),'both_minus_actual_common_denominator_components':gap.tolist(),'both_minus_actual_component_fractions':gp.tolist(),'both_minus_actual_fractions_joint3_ci95':gci.tolist(),'gates':gates,'audit':audit};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),angle_fractions=boot,actual_gap_fractions=gb);print('FINISHED',gates,point,gp,flush=True)
if __name__=='__main__':main()
