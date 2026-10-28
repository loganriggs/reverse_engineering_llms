from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';stem='bilin18_l17_activity_compiled_maps';j=json.loads((A/f'{stem}.json').read_text());z=np.load(A/f'{stem}_rows.npz');e=z['errors'];w=np.load(A/'bilin18_l17_compiled_omitted_maps_rows.npz')['errors'];old=np.load(A/'bilin18_l17_operator_transfer_confirmation_rows.npz');d=old['effects'];idx=j['methods'].index('project_linear_only_64')
def rec(err,src,ix):return 1-err[:,src][:,:,ix,1].sum((1,2))/d[src][:,ix,1].sum()
def contrast(src,lo,mi):
 out=[];gaps=[]
 for ix in [lo,mi]:
  ar=rec(e,src,ix);wr=rec(w,src,ix);out.append(ar[idx]-wr[idx]);gaps.append(ar[idx]-ar[0])
 return np.array(out+gaps)
src=np.arange(24);lo=np.arange(16);mi=np.arange(16,32);point=contrast(src,lo,mi);rng=np.random.default_rng(j['config']['bootstrap_seed']);boot=[]
for _ in range(j['config']['bootstrap_samples']):
 pairs=rng.integers(0,12,12);boot.append(contrast(np.r_[pairs,12+pairs],rng.integers(0,16,16),16+rng.integers(0,16,16)))
boot=np.array(boot);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],1);j.update(primary_names=['activity_minus_weight_low','activity_minus_weight_median','activity_minus_exact_low','activity_minus_exact_median'],primary=point.tolist(),primary_joint4_ci95=ci.tolist(),primary_gates=(ci[:,0]>np.array([.02,.02,-.02,-.02])).tolist(),finite_KL_recovery={name:rec(e,src,ix).tolist() for name,ix in [('low',lo),('median',mi)]},exact_score_replay_max_abs=float(abs(e[0]-w[0]).max()));j['calibration_activity_energy_capture']=j.pop('weight_energy_capture',j.get('calibration_activity_energy_capture'));np.savez_compressed(A/f'{stem}_bootstrap.npz',primary=boot);(A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n');print('primary',point,'CI',ci,'gates',j['primary_gates']);print('capture',j['calibration_activity_energy_capture'])
for i,name in enumerate(j['methods']):print(name,'KL',[j['finite_KL_recovery'][s][i] for s in ['low','median']])
