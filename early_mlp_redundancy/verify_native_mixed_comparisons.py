"""Independently verify scope aggregation and all same-rank comparison arithmetic."""
import json
from collections import defaultdict
from fit_pca_encoder import V,sha

def main():
 count=0;maximum=0.;hashes={};qualified={x['family'] for x in json.loads((V/'whole_task_native_comparison.json').read_text())['summaries'] if x['competent']}
 def check(x,y):
  nonlocal count,maximum
  err=abs(x-y);assert err<=1e-12+1e-12*abs(y),(x,y);maximum=max(maximum,err);count+=1
 for stem in ['native_mixed_factorial','native_mixed_source_factorial']:
  cp=V/(stem+'_comparison.json');rp=V/(stem+'.json');c=json.loads(cp.read_text());r=json.loads(rp.read_text());hashes[cp.name]=sha(cp);hashes[rp.name]=sha(rp);lookup={(x['arm'],x['scope']):x for x in c['summaries']};grouped=defaultdict(list)
  for x in r['rows']:
   for scope in ['all','family:'+x['family'],'format:'+x['family']+':'+x['format']]+(['contrast_qualified'] if x['family'] in qualified and stem=='native_mixed_factorial' else []):grouped[x['arm'],scope,x['region'],x['corner']].append(x)
  for s in c['summaries']:
   for site,region in [('source','changed_input_positions'),('final','final_position')]:
    for corner in ['native_reconstructed','reconstructed_native','reconstructed_reconstructed']:
     rows=grouped[s['arm'],s['scope'],region,corner];byfamily=defaultdict(list)
     for x in rows:byfamily[x['family']].append(x)
     ratios=[sum(x['response_error'] for x in g)/sum(x['target_energy'] for x in g) for g in byfamily.values() if sum(x['target_energy'] for x in g)>0];check(sum(ratios)/len(ratios),s[site+'_'+corner+'_family']);check(sum(x['response_error'] for x in rows)/sum(x['target_energy'] for x in rows),s[site+'_'+corner+'_pooled'])
  for item in c['comparisons']:
   scope=item['scope'];candidate=lookup[item['basis']+'__'+str(item['rank']),scope];euclidean=lookup['euclidean__'+str(item['rank']),scope];shared=lookup['shared_linear',scope];native=lookup['native',scope]
   for k,values in item['metrics'].items():
    for name,source in [('candidate',candidate),('euclidean_same_rank',euclidean),('shared_linear',shared),('native',native)]:check(values[name],source[k])
    check(values['candidate_minus_euclidean'],candidate[k]-euclidean[k]);check(values['candidate_minus_shared'],candidate[k]-shared[k])
 out=dict(status='passed',source_sha256=sha(__file__),inputs=hashes,checks=count,max_absolute_error=maximum,scope='Every scope source/final×threecorner pooled/equalfamily response aggregate fromendpointrecords; all comparison arithmetic verified. Other underlying metrics independently rescored infullvectoraudits.');f=V/'native_mixed_comparison_audit.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
