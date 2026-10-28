from pathlib import Path
import sys,json,numpy as np,torch
r=Path('/home/loganriggs/Coding/neuron_circuits');a=r/'artifacts/bilin18_tensor_decomposition';j=json.loads((a/'bilin18_task_operand_transport.json').read_text());saved=torch.load(a/'bilin18_task_operand_transport_operands.pt',weights_only=False,map_location='cpu');cache=saved['cache'];sys.path.insert(0,str(r/'scripts'))
from audit_bilin18_mlp_discriminators import SNAPSHOT
state=torch.load(SNAPSHOT/'pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu');torch.set_num_threads(6)
checks=[]
for f,l in j['config']['selected_layers'].items():
 w=state[f'transformer.h.{l}.mlp.Down.weight'].to(torch.bfloat16).double()
 for g in j['geometry']:
  if g['family']!=f:continue
  rec=g['recipient'];aa=cache[(g['id'],rec)];bb=cache[(g['id'],1-rec)]
  al,ar,bl,br=[v.double() for v in (aa['Left'],aa['Right'],bb['Left'],bb['Right'])]
  total=w@(bl*br-al*ar);terms=w@((bl-al)*ar+al*(br-ar)+(bl-al)*(br-ar))
  checks.append({'id':g['id'],'recipient':rec,'double_algebra_max_abs':float((total-terms).abs().max()),'float_write_difference_relative_l2':float((total-(bb['write']-aa['write']).double()).norm()/total.norm().clamp_min(1e-20))})
assert max(x['double_algebra_max_abs'] for x in checks)<1e-8
assert max(x['float_write_difference_relative_l2'] for x in checks)<1e-4
assert j['audits']['self_margin_max_abs']<1e-4 and j['audits']['h533_both_replay_margin_max_abs']<1e-4
assert len(j['factorial'])==960 and len(j['transport'])==608
# Pair-cluster bootstrap; both directions within pair remain together.
rng=np.random.default_rng(533534);metrics={}
for f in j['config']['selected_layers']:
 cc=[c for c in j['factorial'] if c['family']==f and c['split']=='transfer'];ids=sorted(set(c['id'] for c in cc));vals={mode:np.array([np.mean([c['donor_directed_delta'] for c in cc if c['id']==i and c['mode']==mode]) for i in ids]) for mode in j['config']['factorial_modes']}
 vals['cross_increment']=vals['both']-vals['additive_without_cross'];bb=rng.integers(0,len(ids),(10000,len(ids)))
 metrics[f]={mode:{'mean':float(v.mean()),'pair_bootstrap_95':np.quantile(v[bb].mean(1),[.025,.975]).tolist()} for mode,v in vals.items()}
 if f in ('agreement','sentiment'):
  cc=[c for c in j['transport'] if c['family']==f];vv={mode:np.array([np.mean([c['donor_directed_delta'] for c in cc if c['id']==i and c['mode']==mode]) for i in ids]) for mode in j['config']['transport_modes']}
  metrics[f]['transport']={mode:{'mean':float(v.mean()),'pair_bootstrap_95':np.quantile(v[bb].mean(1),[.025,.975]).tolist()} for mode,v in vv.items()}
  v=vv['shared']-vv['nuisance_norm_matched'];metrics[f]['shared_minus_nuisance']={'mean':float(v.mean()),'pair_bootstrap_95':np.quantile(v[bb].mean(1),[.025,.975]).tolist()}
result={'audits':{'double_algebra_max_abs':max(x['double_algebra_max_abs'] for x in checks),'float_write_difference_max_relative_l2':max(x['float_write_difference_relative_l2'] for x in checks),'row_counts_verified':True},'metrics':metrics,'numerical_rows':checks,'uncertainty_scope':'Descriptive resampling of eight structured transfer pairs; not independent random population samples. Directions remain clustered. Does not incorporate template selection or discovery uncertainty.'}
(a/'bilin18_task_operand_transport_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['audits'])
for f,m in metrics.items():print(f,'cross increment',m['cross_increment'],'shared-nuisance',m.get('shared_minus_nuisance'))
