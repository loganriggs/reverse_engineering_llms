"""M31 paired intervention-scheme contrasts and component geometry."""
from pathlib import Path
import hashlib,json,statistics
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rp=P/'adaptive_wrapper_results.json';r=json.loads(rp.read_text());assert json.loads((P/'adaptive_wrapper_audit.json').read_text())['result_sha256']==sha(rp);fp=P/'wrapper_crossover_results.json';fixed=json.loads(fp.read_text());assert json.loads((P/'wrapper_crossover_audit.json').read_text())['result_sha256']==sha(fp);contexts=['reference','passage','sample'];models=[];geometry=[]
for kind in ['scalar','block']:
 for seed in [71,72]:
  for mult in [1.,.1]:
   cells={}
   for scheme,data in [('fixed',fixed),('adaptive',r)]:
    cells[scheme]={}
    for context in contexts:
     cells[scheme][context]={label:next(x for x in data['summaries'] if (x['kind'],x['seed'],x['arm'],x['context'],x['family'])==(kind,seed,arm,context,None)) for label,arm in [('initial','initial'),('joint',f'aggregate_gradient:{mult}:live'),('control',f'aggregate_KL:{mult}:live')]}
   ratios={scheme:{c:cells[scheme][c]['joint']['R']/cells[scheme][c]['control']['R'] for c in contexts} for scheme in cells};models.append(dict(kind=kind,seed=seed,multiplier=mult,cells=cells,joint_control_ratios=ratios));print(kind,seed,mult,ratios)
  for context in contexts:
   rows=[x for x in r['records'] if (x['kind'],x['seed'],x['arm'],x['context'])==(kind,seed,'initial',context)];cosines=[x['task_delta_cosine'] for x in rows if x['task_delta_cosine'] is not None];ratios=[(x['adaptive_energy']/x['transported_energy'])**.5 for x in rows if x['transported_energy']>0];geometry.append(dict(kind=kind,seed=seed,context=context,directions=len(rows),both_nonzero=len(cosines),transported_noops=sum(x['transported_energy']==0 for x in rows),adaptive_noops=sum(x['adaptive_energy']==0 for x in rows),noop_flips=sum((x['adaptive_energy']==0)!=(x['transported_energy']==0) for x in rows),group_support_changed_directions=sum(x['group_support_changed_tokens']>0 for x in rows),group_support_changed_tokens=sum(x['group_support_changed_tokens'] for x in rows),task_tokens=sum(x['task_tokens'] for x in rows),median_cosine=statistics.median(cosines) if cosines else None,min_cosine=min(cosines) if cosines else None,median_norm_ratio=statistics.median(ratios) if ratios else None,total_transport_energy=sum(x['transported_energy'] for x in rows),total_adaptive_energy=sum(x['adaptive_energy'] for x in rows)))
counts=[]
for mult in [1.,.1]:
 rows=[x for x in models if x['multiplier']==mult];counts.append(dict(multiplier=mult,by_context={c:dict(fixed_response_wins=sum(x['joint_control_ratios']['fixed'][c]<1 for x in rows),adaptive_response_wins=sum(x['joint_control_ratios']['adaptive'][c]<1 for x in rows),ratio_improved_by_adaptation=sum(x['joint_control_ratios']['adaptive'][c]<x['joint_control_ratios']['fixed'][c] for x in rows),fixed_margin_wins=sum(x['cells']['fixed'][c]['joint']['margin_error_sum']<x['cells']['fixed'][c]['control']['margin_error_sum'] for x in rows),adaptive_margin_wins=sum(x['cells']['adaptive'][c]['joint']['margin_error_sum']<x['cells']['adaptive'][c]['control']['margin_error_sum'] for x in rows)) for c in contexts}))
out=dict(adaptive_sha256=sha(rp),fixed_sha256=sha(fp),models=models,geometry=geometry,counts=counts,scope='Same source/context/candidate/group comparison, changing intervention decoding only. Native effect energies differ; within-scheme joint/control ratios are primary. No new optimization or winner selection. Scheduled duplicates/noops retained, geometric summaries descriptive.');(P/'adaptive_wrapper_analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(counts,indent=2));print('GEOMETRY',json.dumps([x for x in geometry if x['context']!='reference'],indent=2))
