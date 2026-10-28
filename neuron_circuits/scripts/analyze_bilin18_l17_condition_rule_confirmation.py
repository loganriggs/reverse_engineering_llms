from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';stem='bilin18_l17_condition_rule_confirmation';j=json.loads((A/f'{stem}.json').read_text());z=np.load(A/f'{stem}_rows.npz');e=z['errors'];d=z['effects'];assert e.shape==(9,9,24,128,2) and np.isfinite(e).all() and j['solve_gate'] and j['condition_gate'];rec=1-e.sum((2,3))/d.sum((0,1))
def contrast(r,sp):return np.array([r[4,0]-r[3,0],(r[4,1+sp]-r[3,1+sp]).mean(),r[4,0]-r[0,0],(r[4,1+sp]-r[0,1+sp]).mean()])
pt=contrast(rec[:,:,1],np.arange(8));rng=np.random.default_rng(j['config']['bootstrap_seed']);boot=[]
for _ in range(j['config']['bootstrap_samples']):
 pairs=rng.integers(0,12,12);sw=np.bincount(np.r_[pairs,12+pairs],minlength=24);dw=np.bincount(rng.integers(0,128,128),minlength=128);sp=rng.integers(0,8,8);rr=1-np.einsum('cbsi,s,i->cb',e[:,:,:,:,1],sw,dw,optimize=True)/np.einsum('si,s,i->',d[:,:,1],sw,dw,optimize=True);boot.append(contrast(rr,sp))
boot=np.array(boot);radius=np.quantile(abs(boot-pt).max(1),.95);ci=np.stack([pt-radius,pt+radius],1);thresholds=np.array([.02,.02,-.02,-.02]);split=z['split_error'].sum((2,3))/d[:,:,0].sum();norm=z['norms'];spec=z['spectra'];norm_summary=np.stack([np.median(norm,axis=(2,3)),np.quantile(norm,.95,axis=(2,3)),norm.max((2,3))],-1)
j.update(primary_names=['condition_rescue_learned','condition_rescue_random_mean','stabilized64_minus_full_learned','stabilized64_minus_full_random_mean'],primary=pt.tolist(),primary_joint4_ci95=ci.tolist(),primary_thresholds=thresholds.tolist(),primary_gates=(ci[:,0]>thresholds).tolist(),finite_KL_recovery=rec[:,:,1].tolist(),Fisher_recovery=rec[:,:,0].tolist(),norm_quantile_labels=['median','p95','max'],norm_summary=norm_summary.tolist(),included_omitted_error_fractions=split.tolist(),numerical_rank_median=np.median((spec>spec[:,:,:,-1,None]*1e-10).sum(-1),axis=2).tolist())
np.savez_compressed(A/f'{stem}_bootstrap.npz',primary=boot);(A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n');print('primary',pt,'CI',ci,'gates',j['primary_gates'])
for i,c in enumerate(j['conditions']):print(c,'KL',rec[i,0,1],rec[i,1:,1].mean(),'normmax',norm_summary[i,0,2],norm_summary[i,1:,2].max(),'split',split[i,0],split[i,1:].mean(0))
print('median ranks',j['numerical_rank_median'])
