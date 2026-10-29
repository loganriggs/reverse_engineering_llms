"""M112 fixed pooled and equal-family comparisons; no outcome-dependent exclusions."""
import json
import numpy as np
from fit_pca_encoder import V,sha


def main():
 summaries=[];families=[];intact=[];bindings={};counts=[]
 for cohort in ['train','source','whole']:
  path=V/f'continuous_tasks_{cohort}.json';r=json.loads(path.read_text());audit=json.loads((V/f'continuous_tasks_{cohort}_audit.json').read_text());assert audit['status']=='passed' and audit['results_sha256']==sha(path);bindings[path.name]=sha(path)
  counts.append(dict(cohort=cohort,forwards=r['forwards'],exact_replays=r['exact_replays'],responses=len(r['rows']),intact=len(r['intact']),ineligible=r['ineligible']))
  views=['all','qualified5'] if cohort=='whole' else ['all']
  for view in views:
   allowed={x['family'] for x in r['rows']}
   if view=='qualified5':allowed-= {'spatial_inverse','role_binding'};assert len(allowed)==5
   for seed in [71,72]:
    for mode in ['hard','continuous']:
     rr=[x for x in r['intact'] if x['seed']==seed and x['mode']==mode and x['family'] in allowed]
     fs=sorted({x['family'] for x in rr});intact.append(dict(cohort=cohort,view=view,seed=seed,mode=mode,pooled_KL=float(np.mean([x['KL'] for x in rr])),equal_family_KL=float(np.mean([np.mean([x['KL'] for x in rr if x['family']==f]) for f in fs])),top1_agreement=float(np.mean([x['top1_agreement'] for x in rr])),pooled_local_error=sum(x['local_error'] for x in rr)/sum(x['local_energy'] for x in rr)))
    for region in ['changed_input_positions','final_position']:
     for condition in ['hard','continuous','matched_hard']:
      rr=[x for x in r['rows'] if x['seed']==seed and x['region']==region and x['condition']==condition and x['family'] in allowed];fs=sorted({x['family'] for x in rr});assert rr;fr=[]
      for f in fs:
       ss=[x for x in rr if x['family']==f];row=dict(cohort=cohort,view=view,seed=seed,region=region,condition=condition,family=f,count=len(ss))
       for name,num,den in [('response','response_error','teacher_response_energy'),('margin','margin_error','teacher_margin_energy'),('local','local_error','native_delta_energy')]:
        n=sum(x[num] for x in ss);d=sum(x[den] for x in ss);row[name+'_error_sum']=n;row[name+'_target_sum']=d;row[name+'_relative_error']=n/d if d else None
       row['candidate_over_native_energy']=sum(x['candidate_delta_energy'] for x in ss)/sum(x['native_delta_energy'] for x in ss) if sum(x['native_delta_energy'] for x in ss) else None
       fr.append(row);families.append(row)
      summary=dict(cohort=cohort,view=view,seed=seed,region=region,condition=condition,count=len(rr),families=fs)
      for name in ['response','margin','local']:
       n=sum(x[name+'_error_sum'] for x in fr);d=sum(x[name+'_target_sum'] for x in fr);valid=[x[name+'_relative_error'] for x in fr if x[name+'_relative_error'] is not None];summary['pooled_'+name+'_error']=n/d if d else None;summary['equal_family_'+name+'_error']=float(np.mean(valid)) if valid else None
      summary['candidate_over_native_energy']=sum(x['candidate_delta_energy'] for x in rr)/sum(x['native_delta_energy'] for x in rr)
      summaries.append(summary)
 out=dict(status='completed',source_sha256=sha(__file__),inputs=bindings,counts=counts,summaries=summaries,family_rows=families,intact=intact,scope='Relative squared full-logprob response error; zero response is1. Source/final, pooled/equal-family and all7/qualified5 views retained. Magnitude control uses no teacher information. No statistical significance or new-task discovery claims.')
 dest=V/'continuous_task_comparison.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n')
 for cohort,view in [('train','all'),('source','all'),('whole','all'),('whole','qualified5')]:
  for seed in [71,72]:
   for region in ['changed_input_positions','final_position']:
    ss=[x for x in summaries if (x['cohort'],x['view'],x['seed'],x['region'])==(cohort,view,seed,region)];print(cohort,view,seed,region,[(x['condition'],round(x['equal_family_response_error'],4),round(x['pooled_response_error'],4)) for x in ss])
 print('INTACT',intact)
if __name__=='__main__':main()
