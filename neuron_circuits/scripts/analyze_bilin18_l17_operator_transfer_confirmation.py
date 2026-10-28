from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';stem='bilin18_l17_operator_transfer_confirmation';j=json.loads((A/f'{stem}.json').read_text());z=np.load(A/f'{stem}_rows.npz');e=z['errors'];d=z['effects'];assert e.shape==(7,24,32,2) and np.isfinite(e).all()
def rec(src,ix,metric=0):
 ee=e[:,src][:,:,ix,metric].sum((1,2));dd=d[src][:,ix,metric].sum();return 1-ee/ee[6],1-ee/dd
def contrast(src,lo,mi):
 l=rec(src,lo)[0];m=rec(src,mi)[0];return np.array([l[3],m[3],l[1]-l[3],m[1]-m[3]])
src=np.arange(24);lo=np.arange(16);mi=np.arange(16,32);point=contrast(src,lo,mi);rng=np.random.default_rng(j['config']['bootstrap_seed']);boot=[]
for _ in range(j['config']['bootstrap_samples']):
 pairs=rng.integers(0,12,12);boot.append(contrast(np.r_[pairs,12+pairs],rng.integers(0,16,16),16+rng.integers(0,16,16)))
boot=np.array(boot);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],1);j.update(primary_names=['shared16_low','shared16_median','local16_minus_shared16_low','local16_minus_shared16_median'],primary=point.tolist(),primary_joint4_ci95=ci.tolist(),primary_gates=(ci[:,0]>np.array([.9,.9,.05,.05])).tolist(),strata={name:{metric:{'omitted_recovery':rec(src,ix,mt)[0].tolist(),'full_recovery':rec(src,ix,mt)[1].tolist()} for metric,mt in [('Fisher',0),('KL',1)]} for name,ix in [('low',lo),('median',mi)]})
np.savez_compressed(A/f'{stem}_bootstrap.npz',primary=boot);(A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n');print('primary',point,'CI',ci,'gates',j['primary_gates']);print('low KL',j['strata']['low']['KL']);print('median KL',j['strata']['median']['KL'])
