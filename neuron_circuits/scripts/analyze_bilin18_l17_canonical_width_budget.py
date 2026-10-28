from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';s='bilin18_l17_canonical_width_budget';runs={r:json.loads((A/f'{s}_rank{r}.json').read_text()) for r in [64,128]};z={r:np.load(A/f'{s}_rank{r}_rows.npz') for r in [64,128]};assert all(all(j['numerical_gates'].values()) for j in runs.values());cfg=runs[64]['config'];rng=np.random.default_rng(cfg['bootstrap_seed']);ix=rng.integers(0,128,(cfg['bootstrap_samples'],128));recovery={};bootrec={};stats={}
for rank in [64,128]:
 for budget in [8,32]:
  for mode in ['negative','positive']:
   ef=z[rank][mode+'_effects'];e=z[rank][mode+f'_budget{budget}_delta'];v=1-e.sum(0)/ef.sum(0);bv=1-e[ix].sum(1)/ef[ix].sum(1);key=f'r{rank}_b{budget}_{mode}';recovery[key]=float(v[0]);bootrec[key]=bv[:,0];stats[key]={'KL_and_raw_recovery':v.tolist(),'marginal95':np.quantile(bv,[.025,.975],axis=0).T.tolist()}
pt=[];bs=[];names=[]
for mode in ['negative','positive']:
 k1=f'r128_b32_{mode}';k0=f'r64_b32_{mode}';pt.append(recovery[k1]-recovery[k0]);bs.append(bootrec[k1]-bootrec[k0]);names.append('width_gain_'+mode)
for rank in [64,128]:
 for mode in ['negative','positive']:
  k1=f'r{rank}_b32_{mode}';k0=f'r{rank}_b8_{mode}';pt.append(recovery[k1]-recovery[k0]);bs.append(bootrec[k1]-bootrec[k0]);names.append(f'budget_gain_r{rank}_{mode}')
for mode in ['negative','positive']:
 key=f'r128_b32_{mode}';pt.append(recovery[key]);bs.append(bootrec[key]);names.append('r128_b32_absolute_'+mode)
pt=np.array(pt);bs=np.stack(bs,1);se=np.maximum(bs.std(0,ddof=1),1e-12);q=np.quantile((abs(bs-pt)/se).max(1),.95);ci=np.stack([pt-q*se,pt+q*se],1);gates=(ci[:,0]>np.array([.05]*6+[.8]*2)).tolist();interactions={}
for mode in ['negative','positive']:
 value=(recovery[f'r128_b32_{mode}']-recovery[f'r128_b8_{mode}'])-(recovery[f'r64_b32_{mode}']-recovery[f'r64_b8_{mode}']);v=(bootrec[f'r128_b32_{mode}']-bootrec[f'r128_b8_{mode}'])-(bootrec[f'r64_b32_{mode}']-bootrec[f'r64_b8_{mode}']);interactions[mode]={'estimate':value,'marginal95':np.quantile(v,[.025,.975]).tolist()}
j={'experiment':'H472','config':cfg,'runs':runs,'primary_names':names,'primary':pt.tolist(),'primary_joint8_ci95':ci.tolist(),'primary_gates':gates,'recovery':stats,'budget_by_width_interaction':interactions,'unique_completed_documents':12992};(A/f'{s}.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(A/f'{s}_bootstrap.npz',primary=bs);print('primary',pt,'CI',ci,'gates',gates)
for rank in [64,128]:print('rank',rank,'selected',runs[rank]['selected_steps'],'history endpoints',runs[rank]['training_history'][0],runs[rank]['training_history'][-1])
for k,v in stats.items():print(k,v['KL_and_raw_recovery'])
