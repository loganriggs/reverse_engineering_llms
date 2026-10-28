"""M27 objective-matched comparison retaining both fixed norms and all starts."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths={name:P/file for name,file in [('joint_native','finite_candidate_native_fidelity.json'),('control_native','aggregate_kl_native_fidelity.json'),('joint_output','finite_candidate_outputs.json'),('control_output','aggregate_kl_outputs.json'),('joint_fit','finite_gradient_steps.json'),('control_fit','aggregate_kl_training_fit.json')]};data={k:json.loads(p.read_text()) for k,p in paths.items()}
for name,file in [('joint_native','finite_candidate_native_fidelity_audit.json'),('control_native','aggregate_kl_native_fidelity_audit.json'),('joint_output','finite_candidate_outputs_audit.json'),('control_output','aggregate_kl_outputs_audit.json'),('joint_fit','finite_gradient_audit.json'),('control_fit','aggregate_kl_training_fit_audit.json')]:assert json.loads((P/file).read_text())['result_sha256']==sha(paths[name])
causal=[];ordinary=[];training=[]
for kind in ['scalar','block']:
 for seed in [71,72]:
  for mult in [1.,.1]:
   arms={'joint':f'aggregate_gradient:{mult}:live','control':f'aggregate_KL:{mult}:live'};cells={}
   jf=next(x for x in data['joint_fit']['summaries'] if (x['kind'],x['seed'],x['condition'])==(kind,seed,arms['joint']));cf=next(x for x in data['control_fit']['summaries'] if (x['kind'],x['seed'],x['arm'],x['family'])==(kind,seed,arms['control'],None));training.append(dict(kind=kind,seed=seed,multiplier=mult,initial_R=cf['initial_R'],joint_R=jf['task_R'],control_R=cf['R'],joint_K=jf['task_K'],control_K=cf['K']))
   for region in ['prefix','task']:
    cells={}
    for which,arm in arms.items():
     base=[x for x in data[which+'_native']['summaries'] if (x['kind'],x['seed'],x['region'],x['arm'])==(kind,seed,region,arm)];full=next(x for x in base if x['quadrant']=='all' and x['family'] is None);fam=[x for x in base if x['quadrant']=='all' and x['family'] is not None];joint=next(x for x in base if x['quadrant']=='evaluation:evaluation' and x['family'] is None);cells[which]=dict(margin=full['relative_margin_error'],logprob=full['relative_logprob_response_error'],equal_family_margin=sum(x['relative_margin_error'] for x in fam if x['relative_margin_error'] is not None)/sum(x['relative_margin_error'] is not None for x in fam),joint_margin=joint['relative_margin_error'],joint_logprob=joint['relative_logprob_response_error'])
    causal.append(dict(kind=kind,seed=seed,multiplier=mult,region=region,cells=cells))
   for domain,split,template in [('natural','evaluation',None),('original','evaluation',None),('expanded','evaluation','train'),('expanded','train','evaluation'),('expanded','evaluation','evaluation')]:
    cells={}
    for which,arm in arms.items():
     x=next(x for x in data[which+'_output']['summaries'] if (x['kind'],x['seed'],x['arm'],x['domain'],x['split'],x['template_split'],x['gate'],x['family'])==(kind,seed,arm,domain,split,template,'all',None));cells[which]={k:x.get(k) for k in ['mean_KL','last_KL','local_relative_error','top1_agreement','task_top1_rate']}
    ordinary.append(dict(kind=kind,seed=seed,multiplier=mult,domain=domain,lexical_split=split,template_split=template,cells=cells))
# Ensure the comparison uses identical interventions and denominators, not only matching labels.
old={(x['pair_id'],x['endpoint'],x['kind'],x['seed'],x['region'],x['group']):x for x in data['joint_native']['records'] if x['arm']=='initial'}
for x in data['control_native']['records']:
 y=old[x['pair_id'],x['endpoint'],x['kind'],x['seed'],x['region'],x['group']]
 for k in ['delta_energy','native_margin_effect','native_margin_energy','native_logprob_response_energy','noop']:assert x[k]==y[k]
counts=[]
for mult in [1.,.1]:
 rows=[x for x in causal if x['multiplier']==mult];counts.append(dict(multiplier=mult,joint_better_than_control={metric:sum(x['cells']['joint'][metric]<x['cells']['control'][metric] for x in rows) for metric in ['margin','logprob','equal_family_margin','joint_margin','joint_logprob']},training_R_better=sum(x['joint_R']<x['control_R'] for x in training if x['multiplier']==mult)))
out=dict(source_hashes={k:sha(v) for k,v in paths.items()},causal=causal,ordinary=ordinary,training=training,counts=counts,scope='Paired raw-norm matched full-data Euclidean directions with same normalization, differing by inclusion of weighted response objective. Same starts/data/basis; descriptive development result, no semantic-unit or long-run-training claim.');(P/'aggregate_kl_comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(counts,indent=2))
for x in training:print('TRAIN',x)
for x in ordinary:
 if x['domain']=='natural' or (x['domain']=='expanded' and x['lexical_split']==x['template_split']=='evaluation'):print('OUTPUT',x['kind'],x['seed'],x['multiplier'],x['domain'],{k:round(v['mean_KL'],6) for k,v in x['cells'].items()})
