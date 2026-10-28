from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';stem='bilin18_l17_covariance_source_confirmation';j=json.loads((A/f'{stem}.json').read_text());z=np.load(A/f'{stem}_rows.npz');e=z['errors'][...,1];d=z['effects'][...,1]
assert e.shape==(2,3,2,64,128) and d.shape==(2,3,64,128)
assert np.isfinite(e).all() and np.isfinite(d).all() and all(j['numerical_gates'].values())
r=1-e.sum((3,4))/d.sum((2,3))[:,:,None]
def contrasts(v):return np.array([v[0,1,0]-v[0,0,0],v[1,1,0]-v[1,0,0],v[0,1,0]-v[0,2,0],v[1,1,0]-v[1,2,0]])
pt=contrasts(r);rng=np.random.default_rng(j['config']['bootstrap_seed']);boot=[]
for _ in range(j['config']['bootstrap_samples']):
 sw=np.bincount(rng.integers(0,64,64),minlength=64);cw=np.bincount(rng.integers(0,128,128),minlength=128)
 rr=1-np.einsum('afmsi,s,i->afm',e,sw,cw,optimize=True)/np.einsum('afsi,s,i->af',d,sw,cw,optimize=True)[:,:,None];boot.append(contrasts(rr))
boot=np.array(boot);radius=np.quantile(abs(boot-pt).max(1),.95);ci=np.stack([pt-radius,pt+radius],1);gates=[bool(ci[k,0]>-.05 and ci[k,1]<.05) if k<2 else bool(ci[k,0]>.20) for k in range(4)]
cal=np.load(A/f'{stem}_source_audit.npz');b=cal['basis'];center=cal['calibration_center'];audit={}
# Covariance comparison in calibrated source coordinates, preserving all <=128 head directions.
_,_,vh=np.linalg.svd(center,full_matrices=False);q=vh[:128].T;c=center@q;target=c.T@c/(len(c)-1)
for key in ['sources_before_norm','sources']:
 vals=[]
 for ss in cal[key]:
  x=ss@q;x=x-x.mean(0);cov=x.T@x/(len(x)-1)
  vals.append({'centered_covariance_relative_frobenius_error':float(np.linalg.norm(cov-target)/np.linalg.norm(target)),'trace_ratio':float(np.trace(cov)/np.trace(target)),'uncentered_top16_energy_fraction':float(np.square(ss@b).sum()/np.square(ss).sum())})
 audit[key]=vals
j.update(primary_names=['covariance_minus_natural_negative','covariance_minus_natural_positive','covariance_minus_isotropic_negative','covariance_minus_isotropic_positive'],primary=pt.tolist(),primary_joint4_ci95=ci.tolist(),primary_gates=gates,finite_KL_recovery=r.tolist(),source_reconstruction_error_fraction=(z['source_error'].sum(-1)/z['source_energy'].sum(-1)[:,None]).tolist(),effect_KL=d.sum((2,3)).tolist(),covariance_audit=audit)
# Independent pooled scalar extraction avoids mixed-index axis mistakes.
for s in range(2):
 for f in range(3):
  for m in range(2):assert np.isclose(r[s,f,m],1-e[s,f,m].sum()/d[s,f].sum(),rtol=0,atol=1e-12)
np.savez_compressed(A/f'{stem}_bootstrap.npz',primary=boot);(A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n');print('recovery',r,'primary',pt,'CI',ci,'gates',gates,flush=True)
