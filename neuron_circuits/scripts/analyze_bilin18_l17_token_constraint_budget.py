from pathlib import Path
import json, numpy as np
R=Path(__file__).resolve().parents[1]; A=R/'artifacts/bilin18_tensor_decomposition'; stem='bilin18_l17_token_constraint_budget'
j=json.loads((A/f'{stem}.json').read_text()); z=np.load(A/f'{stem}_rows.npz'); e=z['errors']; d=np.load(A/'bilin18_l17_context_projection_transfer_rows.npz')['effects']
assert e.shape==(8,5,24,128,2) and np.isfinite(e).all() and all(j['numerical_gates'].values())
for name,h in j['config']['hashes'].items():
 import hashlib
 assert hashlib.sha256((A/name).read_bytes()).hexdigest()==h
rec=1-e.sum((2,3))/d.sum((0,1))
def contrasts(r,sp):
 return np.array([r[3,0]-r[0,0],(r[3,1+sp]-r[0,1+sp]).mean(),r[3,0]-r[7,0],(r[3,1+sp]-r[7,1+sp]).mean()])
pt=contrasts(rec[:,:,1],np.arange(4)); rng=np.random.default_rng(j['config']['bootstrap_seed']); boot=[]
for _ in range(j['config']['bootstrap_samples']):
 pairs=rng.integers(0,12,12); sw=np.bincount(np.r_[pairs,pairs+12],minlength=24); dw=np.bincount(rng.integers(0,128,128),minlength=128); sp=rng.integers(0,4,4)
 rr=1-np.einsum('cbsi,s,i->cb',e[:,:,:,:,1],sw,dw,optimize=True)/np.einsum('si,s,i->',d[:,:,1],sw,dw,optimize=True)
 boot.append(contrasts(rr,sp))
boot=np.array(boot); radius=np.quantile(abs(boot-pt).max(1),.95); ci=np.stack([pt-radius,pt+radius],1); thresholds=np.array([-.02,-.02,.02,.02])
j.update(primary_names=['top128_minus_full_learned','top128_minus_full_random_mean','top128_minus_random_identities_learned','top128_minus_random_identities_random_mean'],primary=pt.tolist(),primary_joint4_ci95=ci.tolist(),primary_thresholds=thresholds.tolist(),primary_gates=(ci[:,0]>thresholds).tolist(),finite_KL_recovery=rec[:,:,1].tolist(),Fisher_recovery=rec[:,:,0].tolist(),mass_quantile_levels=[0,.25,.5,.75,1],mass_quantiles=np.quantile(z['masses'],[0,.25,.5,.75,1],axis=0).tolist())
np.savez_compressed(A/f'{stem}_bootstrap.npz',primary=boot); (A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n')
print('primary',pt,'CI',ci,'gates',j['primary_gates'])
for name,r,f in zip(j['conditions'],rec[:,:,1],rec[:,:,0]):print(name,'KL learned/random',r[0],r[1:].mean(),'Fisher learned/random',f[0],f[1:].mean())
print('mass quantiles',j['mass_quantiles'])
