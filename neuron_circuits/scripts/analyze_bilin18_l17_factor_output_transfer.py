from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';stem='bilin18_l17_factor_output_transfer';z=np.load(A/f'{stem}_rows.npz');e=z['errors'];d=z['effects'];assert e.shape==(2,48,17,17,3,2) and np.isfinite(e).all();results={}
for name,si in [('all',np.arange(48)),('natural',np.arange(24)),('gaussian',np.arange(24,48))]:
 rec=np.empty((2,2,2,3,2));sums=np.empty((2,2,2,2))
 for ni in range(2):
  for oi,os in enumerate([[0],list(range(1,17))]):
   for di,ds in enumerate([[0],list(range(1,17))]):
    ee=e[ni,si][:,os][:,:,ds].sum((0,1,2));dd=d[ni,si][:,os][:,:,ds].sum((0,1,2));rec[ni,oi,di]=1-ee/dd;sums[ni,oi,di]=dd
 results[name]={'recovery':rec.tolist(),'effect_sums':sums.tolist(),'combination_score':(rec[:,:,:,2]-rec[:,:,:,0]-rec[:,:,:,1]).tolist()}
raw=z['raw_errors'];rd=z['raw_effects'];rawrec=np.stack([1-raw[:,os].sum((0,1))/rd[:,os].sum() for os in [[0],list(range(1,17))]])
j=json.loads((A/f'{stem}.json').read_text());j.update(recovery_axes=['response_length_original_or_common','origin_tail_or_controls','destination_tail_or_controls','core64_added64_union128','Fisher_or_exact_KL'],families=results,raw_recovery_origin_by_group=rawrec.tolist());(A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n');print('all exact KL recovery',np.array(results['all']['recovery'])[...,1]);print('raw recovery',rawrec);print('natural union KL',np.array(results['natural']['recovery'])[:,:,:,2,1]);print('gaussian union KL',np.array(results['gaussian']['recovery'])[:,:,:,2,1])
