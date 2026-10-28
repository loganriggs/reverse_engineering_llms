from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';stem='bilin18_l17_context_projection_transfer';j=json.loads((A/f'{stem}.json').read_text());z=np.load(A/f'{stem}_rows.npz');e=z['errors'];d=z['effects'];assert e.shape==(14,24,128,2) and np.isfinite(e).all()
def recover(src,docs,metric=1):return 1-e[:,src][:,:,docs,metric].sum((1,2))/d[src][:,docs,metric].sum()
def contrast(src,docs,spans):
 rr=recover(src,docs);random=rr[4+spans].mean();return np.array([rr[3]-rr[2],rr[3]-random,random-rr[1]])
src=np.arange(24);docs=np.arange(128);spans=np.arange(4);pt=contrast(src,docs,spans);rng=np.random.default_rng(j['config']['bootstrap_seed']);boot=[]
for _ in range(j['config']['bootstrap_samples']):
 pairs=rng.integers(0,12,12);boot.append(contrast(np.r_[pairs,12+pairs],rng.integers(0,128,128),rng.integers(0,4,4)))
boot=np.array(boot);rad=np.quantile(abs(boot-pt).max(1),.95);ci=np.stack([pt-rad,pt+rad],1);j.update(primary_names=['learned_context_minus_fixed','learned_context_minus_random_mean','random_mean_minus_union_radial'],primary=pt.tolist(),primary_joint3_ci95=ci.tolist(),primary_gates=(ci[:,0]>.02).tolist(),finite_KL_recovery=recover(src,docs).tolist(),Fisher_recovery=recover(src,docs,0).tolist());np.savez_compressed(A/f'{stem}_bootstrap.npz',primary=boot);(A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n');print('primary',pt,'CI',ci,'gates',j['primary_gates']);print(list(zip(j['methods'],j['finite_KL_recovery'])))
