"""Recompute every causal contrast directly from audited per-endpoint errors."""
import json
from collections import defaultdict
from fit_pca_encoder import V,sha

def main():
 checked=0;maximum=0.;inputs={}
 for stem in ['product_dictionary_coverage_factorial','product_dictionary_coverage_source_factorial']:
  rp=V/(stem+'.json');cp=V/(stem+'_comparison.json');r=json.loads(rp.read_text());c=json.loads(cp.read_text());inputs[rp.name]=sha(rp);inputs[cp.name]=sha(cp)
  qualified={x['family'] for x in json.loads((V/'whole_task_native_comparison.json').read_text())['summaries'] if x['competent']}
  def metric(seed,arm,scope,site,corner,aggregation):
   region={'source':'changed_input_positions','final':'final_position'}[site];rows=[x for x in r['rows'] if x['seed']==seed and x['arm']==arm and x['region']==region and x['corner']==corner]
   if scope=='contrast_qualified':rows=[x for x in rows if x['family'] in qualified]
   elif scope.startswith('family:'):rows=[x for x in rows if x['family']==scope.split(':',1)[1]]
   elif scope.startswith('format:'):
    family,fmt=scope[len('format:'):].split(':',1);rows=[x for x in rows if x['family']==family and x['format']==fmt]
   if aggregation=='pooled':return sum(x['response_error'] for x in rows)/sum(x['target_energy'] for x in rows)
   groups=defaultdict(list)
   for x in rows:groups[x['family']].append(x)
   ratios=[sum(x['response_error'] for x in g)/sum(x['target_energy'] for x in g) for g in groups.values() if sum(x['target_energy'] for x in g)>0]
   return sum(ratios)/len(ratios)
  for item in c['interactions']:
   for site in ['source','final']:
    for corner in ['native_reconstructed','reconstructed_native','reconstructed_reconstructed']:
     for agg in ['family','pooled']:
      name=f'{site}_{corner}_{agg}';m=item['metrics'][name];v={coverage:{obj:metric(item['seed'],coverage+'__'+obj,item['scope'],site,corner,agg) for obj in ['intact_KL','joint_three_corner']} for coverage in ['full_pool','old_visited']};f=v['full_pool'];o=v['old_visited'];want=dict(full_joint_minus_KL=f['joint_three_corner']-f['intact_KL'],control_joint_minus_KL=o['joint_three_corner']-o['intact_KL'],interaction=(f['joint_three_corner']-f['intact_KL'])-(o['joint_three_corner']-o['intact_KL']),joint_full_minus_control=f['joint_three_corner']-o['joint_three_corner'],KL_full_minus_control=f['intact_KL']-o['intact_KL'])
      for k,x in want.items():
       error=abs(x-m[k]);assert error<1e-12,(stem,name,k,x,m[k]);maximum=max(maximum,error);checked+=1
 out=dict(status='passed',source_sha256=sha(__file__),inputs=inputs,checked_contrasts=checked,max_absolute_error=maximum,scope='Every source/final ×threecorner×pooled/equalfamily interaction and absolute contrast, allfamilies/formats/cohorts, recomputed from endpoint records. Underlying fullvectors independently audited separately.');p=V/'product_dictionary_coverage_contrast_audit.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
