from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';stem='bilin18_l17_baseline_context_confirmation';j=json.loads((A/f'{stem}.json').read_text());z=np.load(A/f'{stem}_rows.npz');e=z['errors'];d=z['effects'];assert e.shape==(2,12,32,32,3,2) and np.isfinite(e).all()
def agg(ni,src,lo,mi,metric=1):
 out=np.empty((2,2,3))
 for oi,os in enumerate([lo,mi]):
  for di,ds in enumerate([lo,mi]):
   ee=e[ni,src][:,os][:,:,ds,: ,metric].sum((0,1,2));dd=d[ni,src][:,os][:,:,ds,metric].sum();out[oi,di]=1-ee/dd
 return out
def contrast(rec):
 u=rec[:,:,2];dest=u[1,0]-u[1,1];return np.array([rec[0,1,1]-rec[1,1,1],dest,dest-(u[0,0]-u[0,1])])
src=np.arange(12);lo=np.arange(16);mi=np.arange(16,32);rec=agg(0,src,lo,mi);point=contrast(rec);rng=np.random.default_rng(j['config']['bootstrap_seed']);boot=[]
for b in range(j['config']['bootstrap_samples']):
 pairs=rng.integers(0,6,6);ll=rng.integers(0,16,16);mm=16+rng.integers(0,16,16);boot.append(contrast(agg(0,np.r_[pairs,6+pairs],ll,mm)))
boot=np.array(boot);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],1);j.update(primary=point.tolist(),primary_joint3_ci95=ci.tolist(),meaningful05=(ci[:,0]>.05).tolist(),equivalence05=(abs(ci)<.05).all(1).tolist(),sign_positive=(ci[:,0]>0).tolist(),exact_KL_recovery_original=rec.tolist(),exact_KL_recovery_common=agg(1,src,lo,mi).tolist(),Fisher_recovery_original=agg(0,src,lo,mi,0).tolist(),bootstrap_scope='Resample paired source draws and selected documents within frozen low/median strata; reuse the same sampled document indices on origin and destination axes. Conditional on baseline-selected strata and fixed donor pool; selection uncertainty is not resampled.')
np.savez_compressed(A/f'{stem}_bootstrap.npz',primary=boot);(A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n');print('primary',point,'CI',ci,'recovery',rec,'common',np.array(j['exact_KL_recovery_common']),flush=True)
