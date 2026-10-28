from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';stem='bilin18_l17_source_norm_policy';j=json.loads((A/f'{stem}.json').read_text());z=np.load(A/f'{stem}_rows.npz');e=z['errors'][...,1];d=z['effects'][...,1];assert e.shape==(2,4,2,64,128) and d.shape==(2,4,64,128);assert np.isfinite(e).all() and np.isfinite(d).all() and all(j['numerical_gates'].values())
r=1-e.sum((3,4))/d.sum((2,3))[:,:,None]
def contrast(v):return np.array([v[s,f,0]-v[s,ref,0] for ref in [0,1] for f in [2,3] for s in [0,1]])
pt=contrast(r);rng=np.random.default_rng(j['config']['bootstrap_seed']);boot=[]
for _ in range(j['config']['bootstrap_samples']):
 sw=np.bincount(rng.integers(0,64,64),minlength=64);cw=np.bincount(rng.integers(0,128,128),minlength=128);rr=1-np.einsum('afmsi,s,i->afm',e,sw,cw,optimize=True)/np.einsum('afsi,s,i->af',d,sw,cw,optimize=True)[:,:,None];boot.append(contrast(rr))
boot=np.array(boot);radius=np.quantile(abs(boot-pt).max(1),.95);ci=np.stack([pt-radius,pt+radius],1);gates=[bool(ci[k,0]>-.02 and ci[k,1]<.02) if k<4 else bool(ci[k,0]>.02) for k in range(8)]
names=[f'{j["families"][f]}_minus_{j["families"][ref]}_{sign}' for ref in [0,1] for f in [2,3] for sign in ['negative','positive']]
j.update(primary_names=names,primary=pt.tolist(),primary_joint8_ci95=ci.tolist(),primary_gates=gates,finite_KL_recovery=r.tolist(),source_reconstruction_error_fraction=(z['source_error'].sum(-1)/z['source_energy'].sum(-1)[:,None]).tolist(),effect_KL=d.sum((2,3)).tolist(),donor_effect_effective_count=(d.sum((2,3))**2/(d.sum(-1)**2).sum(-1)).tolist())
for s in range(2):
 for f in range(4):
  for m in range(2):assert abs(r[s,f,m]-(1-e[s,f,m].sum()/d[s,f].sum()))<1e-12
np.savez_compressed(A/f'{stem}_bootstrap.npz',primary=boot);(A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n');print('recovery',r,'primary',pt,'CI',ci,'gates',gates,flush=True)
