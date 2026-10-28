from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';stem='bilin18_l17_calibration_coverage';j=json.loads((A/f'{stem}.json').read_text());f=np.load(A/f'{stem}_frozen_bases.npz');assert len(set(f['training_ids'])&set(f['validation_ids']))==0 and len(set(np.r_[f['training_ids'],f['validation_ids']]))==512;sel=j['evaluation_basis_indices'].index(j['overall_selected']);e=np.load(A/f'{stem}_rows.npz')['errors'];w=np.load(A/'bilin18_l17_compiled_omitted_maps_rows.npz')['errors'];d=np.load(A/'bilin18_l17_operator_transfer_confirmation_rows.npz')['effects'];assert np.isfinite(e).all()
def contrast(src,lo,mi):
 out=[];gap=[]
 for ix in [lo,mi]:
  den=d[src][:,ix,1].sum();ae=e[sel,src][:,ix,1].sum();we=w[6,src][:,ix,1].sum();ex=w[0,src][:,ix,1].sum();out.append((we-ae)/den);gap.append((ex-ae)/den)
 return np.array(out+gap)
p=contrast(np.arange(24),np.arange(16),np.arange(16,32));rng=np.random.default_rng(j['config']['bootstrap_seed']);b=[]
for _ in range(j['config']['bootstrap_samples']):
 pairs=rng.integers(0,12,12);b.append(contrast(np.r_[pairs,12+pairs],rng.integers(0,16,16),16+rng.integers(0,16,16)))
b=np.array(b);rad=np.quantile(abs(b-p).max(1),.95);ci=np.stack([p-rad,p+rad],1);j.update(primary_names=['selected_minus_weight_low','selected_minus_weight_median','selected_minus_exact_low','selected_minus_exact_median'],primary=p.tolist(),primary_joint4_ci95=ci.tolist(),primary_gates=(ci[:,0]>np.array([.02,.02,-.02,-.02])).tolist(),split_audit_pass=True);np.savez_compressed(A/f'{stem}_bootstrap.npz',primary=b);(A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n');print('point',p,'CI',ci,'gates',j['primary_gates'])
