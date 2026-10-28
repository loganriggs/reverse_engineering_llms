from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';s='bilin18_l17_canonical_fresh_confirmation';j=json.loads((A/f'{s}.json').read_text());z=np.load(A/f'{s}_rows.npz');assert all(j['numerical_gates'].values());rng=np.random.default_rng(j['config']['bootstrap_seed']);ix=(rng.integers(0,8,(j['config']['bootstrap_samples'],64,8))*64+np.arange(64)[None,:,None]).reshape(-1,512);stats={};boots={};conc={}
def concentration(v):
 return {'largest_case_share':float(v.max()/v.sum()),'top26_cases_share':float(np.sort(v)[-26:].sum()/v.sum()),'effective_case_count':float(v.sum()**2/(v*v).sum())}
for mode in ['negative','positive']:
 ef=z[mode+'_effects'];conc[mode]={'effect_KL':concentration(ef[:,0])}
 for n in ['r64','r128']:
  er=z[mode+'_'+n+'_delta'];rec=1-er.sum(0)/ef.sum(0);b=1-er[ix].sum(1)/ef[ix].sum(1);boots[mode+'_'+n]=b[:,0];stats[mode+'_'+n]={'KL_and_raw_recovery':rec.tolist(),'marginal95':np.quantile(b,[.025,.975],axis=0).T.tolist(),'mean_effect_KL_and_raw':ef.mean(0).tolist()};loo=1-(er[:,0].sum()-er[:,0])/(ef[:,0].sum()-ef[:,0]);conc[mode][n]={'error_KL':concentration(er[:,0]),'leave_one_case_out_KL_recovery_range':[float(loo.min()),float(loo.max())]}
pt=[];bs=[];names=[]
for n in ['r64','r128']:
 for mode in ['negative','positive']:
  key=mode+'_'+n;pt.append(stats[key]['KL_and_raw_recovery'][0]);bs.append(boots[key]);names.append(n+'_absolute_'+mode)
for mode in ['negative','positive']:
 pt.append(stats[mode+'_r128']['KL_and_raw_recovery'][0]-stats[mode+'_r64']['KL_and_raw_recovery'][0]);bs.append(boots[mode+'_r128']-boots[mode+'_r64']);names.append('width_gain_'+mode)
d=z['intact_r128'][:,1]-z['intact_r64'][:,1];pt.append(d.mean());bs.append(d[ix].mean(1));names.append('intactKL_wider_minus_narrow');pt=np.array(pt);bs=np.stack(bs,1);se=np.maximum(bs.std(0,ddof=1),1e-12);q=np.quantile((abs(bs-pt)/se).max(1),.95);ci=np.stack([pt-q*se,pt+q*se],1);gates=(ci[:6,0]>np.array([.8]*4+[.05]*2)).tolist()+[bool(ci[6,1]<.01)];j.update(primary_names=names,primary=pt.tolist(),primary_joint7_ci95=ci.tolist(),primary_gates=gates,delta_prediction=stats,concentration=conc);(A/f'{s}.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(A/f'{s}_bootstrap.npz',primary=bs);print('primary',pt,'CI',ci,'gates',gates);print('delta',stats);print('concentration',conc)
