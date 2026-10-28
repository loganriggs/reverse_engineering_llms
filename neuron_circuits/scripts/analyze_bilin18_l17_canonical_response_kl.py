from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';s='bilin18_l17_canonical_response_kl';j=json.loads((A/f'{s}.json').read_text());z=np.load(A/f'{s}_rows.npz');assert all(j['numerical_gates'].values());rng=np.random.default_rng(j['config']['bootstrap_seed']);ix=rng.integers(0,128,(j['config']['bootstrap_samples'],128));delta={}
for mode in ['negative','positive']:
 ef=z[mode+'_effects']
 for n in ['joint16','response16','kl16']:
  e=z[mode+'_'+n+'_delta'];rec=1-e.sum(0)/ef.sum(0);br=1-e[ix].sum(1)/ef[ix].sum(1);strata={}
  for label,mask in [('trained_source_indices',np.arange(128)%64<32),('untrained_source_indices',np.arange(128)%64>=32)]:strata[label]=(1-e[mask].sum(0)/ef[mask].sum(0)).tolist()
  delta[mode+'_'+n]={'KL_and_raw_recovery':rec.tolist(),'marginal95':np.quantile(br,[.025,.975],axis=0).T.tolist(),'strata':strata}
pt=[];boot=[]
for mode in ['negative','positive']:
 ef=z[mode+'_effects'][:,0];d=z[mode+'_joint16_delta'][:,0]-z[mode+'_kl16_delta'][:,0];pt.append(d.sum()/ef.sum());boot.append(d[ix].sum(1)/ef[ix].sum(1))
d=z['intact_kl16'][:,1]-z['intact_joint16'][:,1];pt.append(d.mean());boot.append(d[ix].mean(1))
for mode in ['negative','positive']:
 ef=z[mode+'_effects'][:,0];e=z[mode+'_kl16_delta'][:,0];pt.append(1-e.sum()/ef.sum());boot.append(1-e[ix].sum(1)/ef[ix].sum(1))
pt=np.array(pt);boot=np.stack(boot,1);se=np.maximum(boot.std(0,ddof=1),1e-12);q=np.quantile((abs(boot-pt)/se).max(1),.95);ci=np.stack([pt-q*se,pt+q*se],1);gates=[bool(ci[0,0]>.1),bool(ci[1,0]>.1),bool(ci[2,1]<.01),bool(ci[3,0]>.8),bool(ci[4,0]>.8)]
j.update(primary_names=['kl16_minus_joint16_deltaKL_negative','kl16_minus_joint16_deltaKL_positive','kl16_minus_joint16_intactKL','kl16_absolute_deltaKL_negative','kl16_absolute_deltaKL_positive'],primary=pt.tolist(),primary_joint5_ci95=ci.tolist(),primary_gates=gates,delta_prediction=delta);(A/f'{s}.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(A/f'{s}_bootstrap.npz',primary=boot);print('primary',pt,'CI',ci,'gates',gates)
for mode in ['negative','positive']:
 for n in ['joint16','response16','kl16']:print(mode,n,delta[mode+'_'+n]['KL_and_raw_recovery'])
print('training selected',j['selected_step']);print([(r['step'],r['training_KL'],r['validation_KL']) for r in j['training_history']])
