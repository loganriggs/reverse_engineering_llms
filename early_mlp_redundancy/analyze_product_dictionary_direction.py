"""M97 finite changes versus fixed endpoint directional slopes; no fitting."""
import json,sys,math
import numpy as np
from fit_pca_encoder import V,sha

def estimate(before,after,slope0,slope1):
 delta=after-before;trap=(slope0+slope1)/2
 return dict(initial=before,final=after,finite_change=delta,slope0=slope0,slope1=slope1,trapezoid=trap,first_order_residual=delta-slope0,trapezoid_residual=delta-trap,first_order_sign_agreement=bool(np.sign(delta)==np.sign(slope0)))

def main():
 cohort=sys.argv[1];assert cohort in ['training','source'];rp=V/f'product_dictionary_direction_{cohort}.json';r=json.loads(rp.read_text());ap=V/f'product_dictionary_direction_{cohort}_audit.json';a=json.loads(ap.read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp);tp=json.loads((V/'balanced_response_training_protocol.json').read_text());den={(x['family'],x['region']):x['denominator'] for x in tp['scales']};lookup={(x['seed'],x['assignment'],x['key'],x['anchor']):x for x in r['rows']};cases=[];branches=[]
 teacher={}
 if cohort=='source':
  source=json.loads((V/'product_dictionary_source_factorial.json').read_text())
  teacher={(x['key'],x['region']):x['target_energy'] for x in source['rows']}
 for (seed,assignment,key,anchor),before in lookup.items():
  if anchor:continue
  after=lookup[seed,assignment,key,1];identity={k:before[k] for k in ['seed','assignment','key','family']};metrics={name:estimate(before[name],after[name],before[name+'_slope'],after[name+'_slope']) for name in ['K','B']};cases.append(identity|dict(metrics=metrics,native_write_energy=before['native_write_energy'],write_direction_energy=before['write_direction_energy'],secant_rounding_error_energy=before['secant_rounding_error_energy']))
  other={(b['region'],b['corner']):b for b in after['branches']};n=sum(x['region']!='identity' for x in before['branches'])
  for b0 in before['branches']:
   b1=other[b0['region'],b0['corner']];region=b0['region'];value=estimate(b0['R'],b1['R'],b0['slope'],b1['slope']);weight=1/(den[before['family'],region]*n) if region!='identity' else 0.
   branches.append(identity|dict(region=region,corner=b0['corner'],metric=value,B_weight=weight,teacher_energy=teacher.get((key,region))))
 summaries=[];branch_summaries=[]
 for seed in [71,72]:
  for family in [None]+sorted({c['family'] for c in cases}):
   rr=[c for c in cases if c['seed']==seed and (family is None or c['family']==family)];out=dict(seed=seed,family=family,n=len(rr),metrics={});energy=math.fsum(x['native_write_energy'] for x in rr);movement=math.fsum(x['write_direction_energy'] for x in rr);out['relative_write_direction_norm']=math.sqrt(movement/energy);out['relative_secant_rounding_error']=math.sqrt(math.fsum(x['secant_rounding_error_energy'] for x in rr)/movement)
   for metric in ['K','B']:
    values=[c['metrics'][metric] for c in rr];out['metrics'][metric]=estimate(*[float(np.mean([x[k] for x in values])) for k in ['initial','final','slope0','slope1']]);total_change=math.fsum(abs(x['finite_change']) for x in values);out['metrics'][metric]['sum_abs_first_order_residual_over_sum_abs_change']=math.fsum(abs(x['first_order_residual']) for x in values)/total_change if total_change else None;out['metrics'][metric]['sum_abs_trapezoid_residual_over_sum_abs_change']=math.fsum(abs(x['trapezoid_residual']) for x in values)/total_change if total_change else None;out['metrics'][metric]['case_sign_agreements']=sum(x['first_order_sign_agreement'] for x in values)
   summaries.append(out)
   for region in ['changed_input_positions','final_position','identity']:
    for corner in ['native_reconstructed','reconstructed_native','reconstructed_reconstructed']:
     bb=[b for b in branches if b['seed']==seed and b['region']==region and b['corner']==corner and (family is None or b['family']==family)];fields=['initial','final','slope0','slope1'];contribution=estimate(*[math.fsum(b['metric'][k]*b['B_weight'] for b in bb)/len(rr) for k in fields]);raw=estimate(*[float(np.mean([b['metric'][k] for b in bb])) for k in fields]) if bb else None;record=dict(seed=seed,family=family,region=region,corner=corner,n=len(bb),all_cases=len(rr),raw_mean=raw,B_contribution=contribution)
     if cohort=='source':
      target=math.fsum(b['teacher_energy'] for b in bb);record['pooled_benchmark']=estimate(*[math.fsum(b['metric'][k] for b in bb)/target for k in fields]) if target else None
      per_family=[]
      for fam in sorted({b['family'] for b in bb}):
       ff=[b for b in bb if b['family']==fam];target=math.fsum(b['teacher_energy'] for b in ff)
       if target:per_family.append([math.fsum(b['metric'][k] for b in ff)/target for k in fields])
      record['equal_family_benchmark']=estimate(*np.mean(per_family,axis=0).tolist()) if per_family else None
     branch_summaries.append(record)
   chosen=[x for x in branch_summaries if x['seed']==seed and x['family']==family]
   for field in ['initial','final','slope0','slope1']:assert np.isclose(sum(x['B_contribution'][field] for x in chosen),out['metrics']['B'][field],rtol=1e-6,atol=1e-8)
 out=dict(status='completed',source_sha256=sha(__file__),result_sha256=sha(rp),audit_sha256=sha(ap),cohort=cohort,cases=cases,branches=branches,summaries=summaries,branch_summaries=branch_summaries,scope='Fixed KL-tojoint path only. Finitechange compared with derivative at0 and trapezoidal endpointderivatives; residuals descriptive, not curvature bounds. Allfamilies/cases preserved; positive trainingden and originalsourcebenchmark normalizations reported separately. No optimization or modelselection.')
 dest=V/f'product_dictionary_direction_{cohort}_comparison.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('ANALYZED',cohort,len(cases),len(branches))
if __name__=='__main__':main()
