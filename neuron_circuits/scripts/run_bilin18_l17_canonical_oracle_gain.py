import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_canonical_oracle_gain'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();old=np.load(ART/'bilin18_l17_canonical_fresh_confirmation_rows.npz');p=torch.tensor(old['pre'],device='cuda');h=torch.tensor(np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz')['sources'][0],device='cuda').float()[np.arange(512)%64];z=np.load(ART/'bilin18_l17_canonical_width_budget_rank128_compiled.npz');z={k:torch.tensor(z[k],device='cuda') for k in z.files}
 def native(x):return ((x@L.T)*(x@R.T))@D.T+bias
 def lp(s):return (30*torch.tanh(F.rms_norm(s,(1152,))@W.T/30)).double().log_softmax(-1)
 def predict(x):
  k=(x.double()@z['support'].T/1152).square();return (z['scale']*((k@z['budget32_read']-k.mean(1,keepdim=True)*z['budget32_rowmean_read'][None])@z['budget32_write']+z['budget32_intercept'])).float()
 x0=F.rms_norm(p,(1152,));y0=native(x0);s=p+y0;v0=predict(x0);saved={};summary={};audit={}
 audit['baseline_relative']=float((s-torch.tensor(old['state'],device='cuda')).norm()/s.norm());assert audit['baseline_relative']<1e-6
 for name,alpha in [('negative',-1),('positive',1)]:
  xx=F.rms_norm(p-alpha*h,(1152,));yy=native(xx);v=predict(xx)-v0;u=yy-y0;rows=[];gains=[];local=[]
  for start in range(0,512,16):
   sl=slice(start,start+16);ss=s[sl];vv=v[sl].double();uu=u[sl].double();th=torch.tanh(F.rms_norm(ss,(1152,)).double()@W.double().T/30);prob0=(30*th).softmax(-1)
   def tangent(t):
    q=(native_drms(ss.double(),t)@W.double().T)*(1-th.square());return q-(prob0*q).sum(-1,keepdim=True)
   tv=tangent(vv);tu=tangent(uu);vvF=(prob0*tv.square()).sum(-1);uvF=(prob0*tv*tu).sum(-1);uuF=(prob0*tu.square()).sum(-1);vvR=vv.square().sum(-1);uvR=(vv*uu).sum(-1);uuR=uu.square().sum(-1);assert float(vvF.min())>0 and float(vvR.min())>0
   gs=torch.stack([torch.ones_like(vvF),(uuR/vvR).sqrt(),uvR/vvR,uvF/vvF],-1);truth=lp(p[sl]+yy[sl]);pr=truth.exp();effect=(pr*(truth-lp(ss))).sum(-1);errs=[]
   for j in range(4):
    delta=(gs[:,j,None]*vv).float();q=lp(ss+delta);errs.append(torch.stack([(pr*(truth-q)).sum(-1),(delta.double()-uu).square().sum(-1)],-1))
   rows.append(torch.stack(errs,1).cpu().numpy());gains.append(gs.cpu().numpy());local.append(torch.stack([vvF,uvF,uuF,uvF/(vvF*uuF).sqrt(),effect,uuR],-1).cpu().numpy())
  rr=np.concatenate(rows);gg=np.concatenate(gains);ll=np.concatenate(local);saved[name+'_errors']=rr;saved[name+'_gains']=gg;saved[name+'_local']=ll
  replay=float(np.max(np.abs(rr[:,0]-old[name+'_r128_delta'])));audit[name+'_original_maxabs']=replay;assert np.allclose(rr[:,0],old[name+'_r128_delta'],rtol=2e-5,atol=1e-8)
  # Analytic local optimum must never increase local quadratic error.
  original=ll[:,2]-2*ll[:,1]+ll[:,0];optimal=ll[:,2]-ll[:,1]**2/ll[:,0];assert np.all(optimal<=original+1e-10)
  summary[name]={m:{'KL_recovery':float(1-rr[:,j,0].sum()/ll[:,4].sum()),'raw_recovery':float(1-rr[:,j,1].sum()/ll[:,5].sum()),'gain_quantiles':np.quantile(gg[:,j],[0,.1,.5,.9,1]).tolist(),'finite_KL_worse_fraction':float(np.mean(rr[:,j,0]>rr[:,0,0]))} for j,m in enumerate(cfg['methods'])};summary[name]['fisher_cosine_quantiles']=np.quantile(ll[:,3],[0,.1,.5,.9,1]).tolist();summary[name]['local_original_error']=float(original.sum());summary[name]['local_oracle_error']=float(optimal.sum());print(name,summary[name],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H475','config':cfg,'summary':summary,'audits':audit,'local_columns':['vvF','uvF','uuF','cosine','effect_KL','raw_effect_energy'],'error_columns':['KL','raw_squared_error'],'unique_completed_documents':13504},indent=2)+'\n')
if __name__=='__main__':main()
