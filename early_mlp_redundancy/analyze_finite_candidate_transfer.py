"""M26 all-candidate paired transfer summaries without selection."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
np=P/'finite_candidate_native_fidelity.json';op=P/'finite_candidate_outputs.json';native=json.loads(np.read_text());outputs=json.loads(op.read_text());assert json.loads((P/'finite_candidate_native_fidelity_audit.json').read_text())['result_sha256']==sha(np);assert json.loads((P/'finite_candidate_outputs_audit.json').read_text())['result_sha256']==sha(op);protocol=json.loads((P/'finite_candidate_protocol.json').read_text());causal=[];ordinary=[]
for kind in ['scalar','block']:
 for seed in [71,72]:
  for region in ['prefix','task']:
   cells={}
   for arm in ['initial']+protocol['arms']:
    sums=[x for x in native['summaries'] if (x['kind'],x['seed'],x['region'],x['arm'])==(kind,seed,region,arm)];allrow=next(x for x in sums if x['quadrant']=='all' and x['family'] is None);fam=[x for x in sums if x['quadrant']=='all' and x['family'] is not None];joint=next(x for x in sums if x['quadrant']=='evaluation:evaluation' and x['family'] is None);cells[arm]=dict(margin=allrow['relative_margin_error'],logprob=allrow['relative_logprob_response_error'],equal_family_margin=sum(x['relative_margin_error'] for x in fam if x['relative_margin_error'] is not None)/sum(x['relative_margin_error'] is not None for x in fam),joint_margin=joint['relative_margin_error'],joint_logprob=joint['relative_logprob_response_error'])
   causal.append(dict(kind=kind,seed=seed,region=region,cells=cells))
  for domain,split,template in [('natural','evaluation',None),('original','evaluation',None),('expanded','evaluation','train'),('expanded','train','evaluation'),('expanded','evaluation','evaluation')]:
   cells={}
   for arm in ['initial']+protocol['arms']:
    x=next(x for x in outputs['summaries'] if (x['kind'],x['seed'],x['arm'],x['domain'],x['split'],x['template_split'],x['gate'],x['family'])==(kind,seed,arm,domain,split,template,'all',None));cells[arm]={k:x.get(k) for k in ['mean_KL','last_KL','local_relative_error','top1_agreement','task_top1_rate']}
   ordinary.append(dict(kind=kind,seed=seed,domain=domain,lexical_split=split,template_split=template,cells=cells))
counts=[]
for arm in protocol['arms']:
 counts.append(dict(arm=arm,relative_to_initial={metric:sum(x['cells'][arm][metric]<x['cells']['initial'][metric] for x in causal) for metric in ['margin','logprob','equal_family_margin','joint_margin','joint_logprob']}))
paired=[]
for mult in [1.,.1]:
 a=f'aggregate_gradient:{mult}:live';b=f'minibatch_adam:{mult}:live';paired.append(dict(multiplier=mult,aggregate_better_than_matched_minibatch={metric:sum(x['cells'][a][metric]<x['cells'][b][metric] for x in causal) for metric in ['margin','logprob','equal_family_margin','joint_margin','joint_logprob']}))
out=dict(native_sha256=sha(np),outputs_sha256=sha(op),causal=causal,ordinary=ordinary,counts=counts,paired_counts=paired,scope='All four prespecified candidates per start; descriptive counts of correlated model/region cases, not independent significance tests. Training/evaluation sources familiar development data; no candidate selected.');(P/'finite_candidate_transfer_analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(counts=counts,paired=paired),indent=2))
for x in ordinary:
 if x['domain']=='natural' or (x['domain']=='expanded' and x['lexical_split']==x['template_split']=='evaluation'):print(x['kind'],x['seed'],x['domain'],{k:round(v['mean_KL'],6) for k,v in x['cells'].items()})
