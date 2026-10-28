from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';stem='bilin18_l17_context_projection';j=json.loads((A/f'{stem}.json').read_text());z=np.load(A/f'{stem}_rows.npz');e=z['errors'];d=np.load(A/'bilin18_l17_operator_transfer_confirmation_rows.npz')['effects']
def rec(src,ix,metric=1):return 1-e[:,src][:,:,ix,metric].sum((1,2))/d[src][:,ix,metric].sum()
def contrast(src,lo,mi):
 l=rec(src,lo);m=rec(src,mi);return np.array([l[1]-l[0],m[1]-m[0],l[1]-l[3],m[1]-m[3]])
src=np.arange(24);lo=np.arange(16);mi=np.arange(16,32);pt=contrast(src,lo,mi);rng=np.random.default_rng(j['config']['bootstrap_seed']);b=[]
for _ in range(j['config']['bootstrap_samples']):
 pairs=rng.integers(0,12,12);b.append(contrast(np.r_[pairs,12+pairs],rng.integers(0,16,16),16+rng.integers(0,16,16)))
b=np.array(b);rad=np.quantile(abs(b-pt).max(1),.95);ci=np.stack([pt-rad,pt+rad],1);j.update(primary_names=['context_minus_fixed_low','context_minus_fixed_median','learned_minus_random_context_low','learned_minus_random_context_median'],primary=pt.tolist(),primary_joint4_ci95=ci.tolist(),primary_gates=(ci[:,0]>.02).tolist(),finite_KL_recovery={name:rec(src,ix).tolist() for name,ix in [('low',lo),('median',mi)]},Fisher_recovery={name:rec(src,ix,0).tolist() for name,ix in [('low',lo),('median',mi)]},linear_prediction_norm_ratio_quantiles=np.quantile(z['normratios'],[.5,.9,.99,1],axis=(1,2)).T.tolist());np.savez_compressed(A/f'{stem}_bootstrap.npz',primary=b);(A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n');print('primary',pt,'CI',ci,'gates',j['primary_gates']);print('KL',j['finite_KL_recovery']);print('Fisher',j['Fisher_recovery']);print('norm quantiles',j['linear_prediction_norm_ratio_quantiles'])
