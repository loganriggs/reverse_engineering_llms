from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';stem='bilin18_l17_metric_compiled_selection';j=json.loads((A/f'{stem}.json').read_text());e=np.load(A/f'{stem}_rows.npz')['errors'];ex=np.load(A/'bilin18_l17_compiled_omitted_maps_rows.npz')['errors'][0];d=np.load(A/'bilin18_l17_operator_transfer_confirmation_rows.npz')['effects'];old=np.load(A/'bilin18_l17_calibration_coverage_rows.npz')['errors'];assert np.allclose(e[0],old[2],rtol=1e-11,atol=1e-12)
def contrast(src,lo,mi):
 out=[];gap=[]
 for ix in [lo,mi]:
  den=d[src][:,ix,1].sum();ee=e[:,src][:,:,ix,1].sum((1,2));exact=ex[src][:,ix,1].sum();out.append((ee[0]-ee[1])/den);gap.append((exact-ee[1])/den)
 return np.array(out+gap)
p=contrast(np.arange(24),np.arange(16),np.arange(16,32));rng=np.random.default_rng(j['config']['bootstrap_seed']);b=[]
for _ in range(j['config']['bootstrap_samples']):
 pairs=rng.integers(0,12,12);b.append(contrast(np.r_[pairs,12+pairs],rng.integers(0,16,16),16+rng.integers(0,16,16)))
b=np.array(b);rad=np.quantile(abs(b-p).max(1),.95);ci=np.stack([p-rad,p+rad],1);j.update(primary_names=['selected_minus_unweighted_low','selected_minus_unweighted_median','selected_minus_exact_low','selected_minus_exact_median'],primary=p.tolist(),primary_joint4_ci95=ci.tolist(),primary_gates=(ci[:,0]>np.array([.02,.02,-.02,-.02])).tolist(),unweighted_score_replay_max_abs=float(abs(e[0]-old[2]).max()));np.savez_compressed(A/f'{stem}_bootstrap.npz',primary=b);(A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n');print('point',p,'CI',ci,'gates',j['primary_gates'])
