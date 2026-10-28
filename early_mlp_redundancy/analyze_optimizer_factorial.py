"""M23 two-objective/two-optimizer paired comparisons; no selections or fits."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths={beta:{name:P/(prefix+suffix) for name,suffix in [('training','training_results.json'),('native','native_fidelity.json'),('fit','training_fit.json')]} for beta,prefix in [('0.9','response_'),('0.0','nomomentum_')]};data={b:{k:json.loads(p.read_text()) for k,p in sources.items()} for b,sources in paths.items()};causal=[];training=[];ordinary=[]
for kind in ['scalar','block']:
 for seed in [71,72]:
  for region in ['prefix','task']:
   cells={}
   for beta in ['0.9','0.0']:
    for arm in ['intact_KL','intact_KL_plus_response']:
     source=data[beta]['native'];base=next(x for x in source['summaries'] if (x['kind'],x['seed'],x['region'],x['arm'],x['quadrant'],x['family'])==(kind,seed,region,arm,'all',None));families=[x for x in source['summaries'] if (x['kind'],x['seed'],x['region'],x['arm'],x['quadrant'])==(kind,seed,region,arm,'all') and x['family'] is not None];valid=[x['relative_margin_error'] for x in families if x['relative_margin_error'] is not None];logvalid=[x['relative_logprob_response_error'] for x in families if x['relative_logprob_response_error'] is not None];both=next(x for x in source['summaries'] if (x['kind'],x['seed'],x['region'],x['arm'],x['quadrant'],x['family'])==(kind,seed,region,arm,'evaluation:evaluation',None));cells[beta+':'+arm]=dict(margin=base['relative_margin_error'],logprob=base['relative_logprob_response_error'],equal_family_margin=sum(valid)/len(valid),equal_family_logprob=sum(logvalid)/len(logvalid),nonzero_margin_families=len(valid),both_withheld_margin=both['relative_margin_error'],both_withheld_logprob=both['relative_logprob_response_error'])
   interaction={key:(cells['0.0:intact_KL_plus_response'][key]-cells['0.0:intact_KL'][key])-(cells['0.9:intact_KL_plus_response'][key]-cells['0.9:intact_KL'][key]) for key in ['margin','logprob','equal_family_margin','both_withheld_margin']};causal.append(dict(kind=kind,seed=seed,region=region,cells=cells,response_objective_difference_in_differences=interaction))
  fitcells={}
  for beta in ['0.9','0.0']:
   for arm in ['intact_KL','intact_KL_plus_response']:
    x=next(x for x in data[beta]['fit']['summaries'] if (x['kind'],x['seed'],x['arm'],x['family'])==(kind,seed,arm,None));fitcells[beta+':'+arm]={k:x[k] for k in ['K','R','combined','initial_K','initial_R']}
  training.append(dict(kind=kind,seed=seed,cells=fitcells))
  for domain,part,template in [('natural','evaluation',None),('original','evaluation',None),('expanded','evaluation','train'),('expanded','train','evaluation'),('expanded','evaluation','evaluation')]:
   cells={}
   for beta in ['0.9','0.0']:
    for arm in ['intact_KL','intact_KL_plus_response']:
     x=next(x for x in data[beta]['training']['summaries'] if (x['kind'],x['seed'],x['arm'],x['domain'],x['split'],x['template_split'],x['gate'],x['family'])==(kind,seed,arm,domain,part,template,'all',None));cells[beta+':'+arm]={k:x[k] for k in ['mean_KL','last_KL','local_relative_error']};cells[beta+':'+arm]['task_top1_rate']=x.get('task_top1_rate')
   ordinary.append(dict(kind=kind,seed=seed,domain=domain,lexical_split=part,template_split=template,cells=cells))
# Native denominators and deltas must be identical across optimizer comparisons.
old={(x['pair_id'],x['endpoint'],x['kind'],x['seed'],x['region'],x['group']):x for x in data['0.9']['native']['records'] if x['arm']=='initial'}
for x in data['0.0']['native']['records']:
 y=old[x['pair_id'],x['endpoint'],x['kind'],x['seed'],x['region'],x['group']]
 for k in ['native_margin_energy','native_margin_effect','native_logprob_response_energy','delta_energy','noop']:assert x[k]==y[k]
out=dict(scope='Fixed factorial; paired withinstart objective and optimizer comparisons. Interaction=(response-control at beta0)-(response-control at beta.9); negative favors response objective more under beta0. Descriptive two seeds/two architectures, not independent significance tests. Real optimizer step magnitudes differ.',source_hashes={b:{k:sha(p) for k,p in sources.items()} for b,sources in paths.items()},training_fit=training,causal=causal,ordinary=ordinary);(P/'optimizer_factorial.json').write_text(json.dumps(out,indent=2)+'\n')
for x in training:print('TRAIN',x['kind'],x['seed'],{k:round(v['R'],6) for k,v in x['cells'].items()})
for x in causal:print('CAUSAL',x['kind'],x['seed'],x['region'],{k:(round(v['margin'],3),round(v['logprob'],3)) for k,v in x['cells'].items()})
