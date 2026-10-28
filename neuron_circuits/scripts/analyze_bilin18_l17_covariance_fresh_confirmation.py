from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';stem='bilin18_l17_covariance_fresh_confirmation';j=json.loads((A/f'{stem}.json').read_text());z=np.load(A/f'{stem}_rows.npz');e=z['errors'][...,1];d=z['effects'][...,1];assert e.shape==(2,9,2,64,128) and d.shape==(2,9,64,128);assert np.isfinite(e).all() and np.isfinite(d).all() and all(j['numerical_gates'].values())
en=e[:,0];dn=d[:,0];eg=np.stack([np.concatenate([e[:,g] for g in range(start,start+4)],axis=2) for start in [1,5]],axis=1);dg=np.stack([np.concatenate([d[:,g] for g in range(start,start+4)],axis=1) for start in [1,5]],axis=1)
assert eg.shape==(2,2,2,256,128) and dg.shape==(2,2,256,128)
rn=1-en.sum((2,3))/dn.sum((1,2))[:,None];rg=1-eg.sum((3,4))/dg.sum((2,3))[:,:,None];r=np.concatenate([rn[:,None],rg],axis=1)
def contrast(v):return np.array([v[s,f,0]-v[s,0,0] for f in [1,2] for s in [0,1]]+[v[s,f,0] for f in [1,2] for s in [0,1]])
pt=contrast(r);rng=np.random.default_rng(j['config']['bootstrap_seed']);boot=[]
for _ in range(j['config']['bootstrap_samples']):
 nw=np.bincount(rng.integers(0,64,64),minlength=64);gw=np.bincount(rng.integers(0,256,256),minlength=256);cw=np.bincount(rng.integers(0,128,128),minlength=128)
 nr=1-np.einsum('amsi,s,i->am',en,nw,cw,optimize=True)/np.einsum('asi,s,i->a',dn,nw,cw,optimize=True)[:,None];gr=1-np.einsum('afmsi,s,i->afm',eg,gw,cw,optimize=True)/np.einsum('afsi,s,i->af',dg,gw,cw,optimize=True)[:,:,None];boot.append(contrast(np.concatenate([nr[:,None],gr],axis=1)))
boot=np.array(boot);se=np.maximum(boot.std(0,ddof=1),1e-12);q=np.quantile((abs(boot-pt)/se).max(1),.95);ci=np.stack([pt-q*se,pt+q*se],1);gates=[bool(ci[k,0]>-.02 and ci[k,1]<.02) if k<4 else bool(ci[k,0]>.98) for k in range(8)]
names=[f'{f}_minus_natural_{s}' for f in ['raw','scaled'] for s in ['negative','positive']]+[f'{f}_absolute_{s}' for f in ['raw','scaled'] for s in ['negative','positive']]
ener=z['source_energy'];err=z['source_error'];sr=[]
for ids in [[0],list(range(1,5)),list(range(5,9))]:sr.append([float(err[ids,m,:].sum()/ener[ids,:].sum()) for m in range(2)])
# Scalar recomputation independently checks all method/group aggregation.
for s in range(2):
 for f,ids in enumerate([[0],list(range(1,5)),list(range(5,9))]):
  for m in range(2):
   es=sum(e[s,g,m].sum() for g in ids);ds=sum(d[s,g].sum() for g in ids);assert abs(r[s,f,m]-(1-es/ds))<1e-12
j.update(primary_names=names,primary=pt.tolist(),primary_joint8_ci95=ci.tolist(),primary_gates=gates,bootstrap_standard_errors=se.tolist(),standardized_critical_value=float(q),finite_KL_recovery=r.tolist(),source_reconstruction_error_fraction=sr,effect_KL_mean_per_case=np.concatenate([dn.mean((1,2))[:,None],dg.mean((2,3))],axis=1).tolist())
np.savez_compressed(A/f'{stem}_bootstrap.npz',primary=boot);(A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n');print('recovery',r,'primary',pt,'CI',ci,'gates',gates,flush=True)
