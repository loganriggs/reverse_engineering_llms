from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';s='bilin18_l17_canonical_objective_transfer';j=json.loads((A/f'{s}.json').read_text());z=np.load(A/f'{s}_rows.npz');assert all(j['numerical_gates'].values());w=z['intact_write'];c=z['intact_ce'];k=z['intact_kl'];v=np.stack([c[:,2]-w[:,2],c[:,1]-w[:,1],k[:,1]-w[:,1],c[:,0]-w[:,0]],1);pt=v.mean(0);rng=np.random.default_rng(j['config']['bootstrap_seed']);ix=rng.integers(0,128,(j['config']['bootstrap_samples'],128));boot=v[ix].mean(1);se=np.maximum(boot.std(0,ddof=1),1e-12);q=np.quantile((abs(boot-pt)/se).max(1),.95);ci=np.stack([pt-q*se,pt+q*se],1);gates=[bool(ci[0,1]<0),bool(ci[1,0]>0),bool(ci[2,1]<0),bool(ci[3,0]>0)];delta={}
for mode in ['negative','positive']:
 effects=z[mode+'_effects']
 for n in ['write','kl','ce']:
  er=z[mode+'_'+n+'_delta'];rec=1-er.sum(0)/effects.sum(0);br=1-er[ix].sum(1)/effects[ix].sum(1);delta[mode+'_'+n]={'KL_and_raw_recovery':rec.tolist(),'marginal95':np.quantile(br,[.025,.975],axis=0).T.tolist(),'mean_effect_KL_and_raw':effects.mean(0).tolist()}
j.update(primary_names=['CE_update_minus_write_CE','CE_update_minus_write_KL','KL_update_minus_write_KL','CE_update_minus_write_writeMSE'],primary=pt.tolist(),primary_joint4_ci95=ci.tolist(),primary_gates=gates,delta_prediction=delta);(A/f'{s}.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(A/f'{s}_bootstrap.npz',primary=boot);print('primary',pt,'CI',ci,'gates',gates);print('delta',json.dumps(delta,indent=2))
