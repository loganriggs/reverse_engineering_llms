"""M62 retain all paired arms, common starts, and metric tradeoffs."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 data={}
 for name in ['balanced_eval_train','balanced_eval_holdout','balanced_eval_natural']:
  path=V/(name+'.json');d=json.loads(path.read_text());audit=json.loads((V/(name+'_audit.json')).read_text());assert audit['status']=='passed' and audit['result_sha256']==sha(path);data[name]=d
 oldtrain=json.loads((V/'training_coverage_featurizers.json').read_text());oldhold=json.loads((V/'source_holdout_featurizers.json').read_text());oldnat=json.loads((V/'joint_shared_outputs.json').read_text());p=json.loads((V/'balanced_response_training_protocol.json').read_text());rows=[]
 for initial in p['models']:
  for arm in ['initial']+p['arms']:
   kind,seed=initial['kind'],initial['seed'];metrics={}
   for cohort,old in [('train',oldtrain),('holdout',oldhold)]:
    d=old if arm=='initial' else data['balanced_eval_'+cohort];a='intact_KL' if arm=='initial' else arm
    for region in ['changed_input_positions','final_position']:
     summary=next(s for s in d['summaries'] if s['kind']==kind and s['seed']==seed and s['arm']==a and s['region']==region and s['family'] is None);macro=next(s for s in d['macros'] if s['kind']==kind and s['seed']==seed and s['arm']==a and s['region']==region);prefix=cohort+'_'+('source' if region=='changed_input_positions' else 'final');metrics[prefix+'_pooled']=summary['relative_response_error'];metrics[prefix+'_family_mean']=macro['equal_family_response_error'];metrics[prefix+'_margin_pooled']=summary['relative_margin_error']
    baselines=[b for b in d['baselines'] if b['kind']==kind and b['seed']==seed and b['arm']==a];metrics[cohort+'_intact_KL']=sum(b['final_KL'] for b in baselines)/len(baselines);metrics[cohort+'_task_top1']=sum(b['top1'] for b in baselines)/len(baselines)
   if arm=='initial':
    rs=[r for r in oldnat['rows'] if r['kind']==kind and r['seed']==seed and r['arm']=='intact_KL' and r['domain']=='natural'];assert len(rs)==32
    metrics['natural_KL']=sum(r['mean_KL'] for r in rs)/32;metrics['natural_delta_CE']=sum(r['observed_next_token_dCE'] for r in rs)/32;metrics['natural_active_mean']=sum(r['active_mean'] for r in rs)/32
   else:
    s=next(s for s in data['balanced_eval_natural']['summaries'] if s['kind']==kind and s['seed']==seed and s['arm']==arm);metrics.update(natural_KL=s['mean_KL'],natural_delta_CE=s['delta_CE'],natural_active_mean=s['active_mean'])
   rows.append(dict(kind=kind,seed=seed,arm=arm,metrics=metrics))
 comparisons=[]
 for left,right in [('KL_raw_response','intact_KL'),('KL_balanced_response','intact_KL'),('KL_balanced_response','KL_raw_response')]:
  for metric in rows[0]['metrics']:
   if metric.endswith('active_mean'):continue
   pairs=[]
   for initial in p['models']:
    a=next(r for r in rows if r['kind']==initial['kind'] and r['seed']==initial['seed'] and r['arm']==left)['metrics'][metric];b=next(r for r in rows if r['kind']==initial['kind'] and r['seed']==initial['seed'] and r['arm']==right)['metrics'][metric];pairs.append(dict(kind=initial['kind'],seed=initial['seed'],left=a,right=b,difference=a-b,left_better=a>b if metric.endswith('top1') else a<b))
   comparisons.append(dict(left=left,right=right,metric=metric,wins=sum(x['left_better'] for x in pairs),comparisons=6,pairs=pairs))
 op=V/'balanced_eval_comparison.json';assert not op.exists();op.write_text(json.dumps(dict(status='completed',source_sha256=sha(Path(__file__)),sources={n:sha(V/(n+'.json')) for n in data},initial_sources={n:sha(V/(n+'.json')) for n in ['training_coverage_featurizers','source_holdout_featurizers','joint_shared_outputs']},rows=rows,comparisons=comparisons,scope='Every final arm plus same six initial M36KL states. Descriptive paired two-seed comparisons, no winner selection. Holdouts cover six familiar development families; natural retention separate. Old initial natural CE and new CE share observed-next-token definition.'),indent=2)+'\n')
 for c in comparisons:
  if c['left']=='KL_balanced_response' and c['right']=='KL_raw_response' and c['metric'] in ['train_source_family_mean','train_final_family_mean','holdout_source_family_mean','holdout_final_family_mean','natural_KL','holdout_intact_KL']:print(c['metric'],c['wins'],'/6')
if __name__=='__main__':main()
