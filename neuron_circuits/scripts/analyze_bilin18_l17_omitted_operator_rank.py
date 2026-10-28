from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';stem='bilin18_l17_omitted_operator_rank';j=json.loads((A/f'{stem}.json').read_text());z=np.load(A/f'{stem}_rows.npz');err=z['errors'];old=np.load(A/'bilin18_l17_baseline_context_confirmation_rows.npz');base=np.stack([old['errors'][0,:,i,i,2] for i in range(32)],1);effect=np.stack([old['effects'][0,:,i,i] for i in range(32)],1)
def rec(src,ix,metric=0):
 ee=err[:,:,src][:,:,:,ix,metric].sum((2,3));bb=base[src][:,ix,metric].sum();ff=effect[src][:,ix,metric].sum();return 1-ee/bb,1-ee/ff
def primary(src,lo,mi):
 low=rec(src,lo)[0][:,1];med=rec(src,mi)[0][:,1];return np.array([low[0],med[0],low[0]-low[2],med[0]-med[2]])
src=np.arange(12);lo=np.arange(16);mi=np.arange(16,32);point=primary(src,lo,mi);rng=np.random.default_rng(j['config']['bootstrap_seed']);boot=[]
for _ in range(j['config']['bootstrap_samples']):
 pairs=rng.integers(0,6,6);boot.append(primary(np.r_[pairs,6+pairs],rng.integers(0,16,16),16+rng.integers(0,16,16)))
boot=np.array(boot);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],1);j.update(primary_names=['rank16_local_low','rank16_local_median','rank16_local_minus_shared_low','rank16_local_minus_shared_median'],primary=point.tolist(),primary_joint4_ci95=ci.tolist(),primary_gates=(ci[:,0]>np.array([.9,.9,.05,.05])).tolist(),strata={name:{metric:{'omitted_recovery':rec(src,ix,mt)[0].tolist(),'full_recovery':rec(src,ix,mt)[1].tolist()} for metric,mt in [('Fisher',0),('KL',1)]} for name,ix in [('low',lo),('median',mi)]})
np.savez_compressed(A/f'{stem}_bootstrap.npz',primary=boot);(A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n');print('primary',point,'CI',ci,'gates',j['primary_gates']);print('low fullKL',np.array(j['strata']['low']['KL']['full_recovery']));print('median fullKL',np.array(j['strata']['median']['KL']['full_recovery']))
