"""Post-result M16 localization of prediction error; descriptive, no repair."""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent/'validation';ip=P/'context_region_analysis.json';d=json.loads(ip.read_text());rows=[]
for r in d['results']:
 if r['kind']!='block' or r['grouping']!='contiguous':continue
 stem=r['context']+':block:contiguous:';a=d['matrices'][stem+'71:'+r['region']];b=d['matrices'][stem+'72:'+r['region']];target=[b[i] for i in r['assignment']];errors=[sum((r['amplitudes'][i]*x-y)**2 for x,y in zip(a[i],target[i])) for i in range(32)];total=sum(errors);i=max(range(32),key=errors.__getitem__);energy=sum(x*x for row in target for x in row);assert abs(total/energy-r['score']['relative_squared_error'])<1e-10
 rows.append(dict(context=r['context'],region=r['region'],method=r['method'],largest_source_group=i,target_group=r['assignment'][i],amplitude=r['amplitudes'][i],largest_error_share=errors[i]/total,source15_error_share=errors[15]/total,source15_effect_norm=sum(x*x for x in a[15])**.5,target_for_source15_effect_norm=sum(x*x for x in target[15])**.5))
out=dict(scope='Post-result error accounting, no refitting, exclusion or confirmation.',analysis_sha256=hashlib.sha256(ip.read_bytes()).hexdigest(),rows=rows);(P/'context_region_calibration.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:
 if r['method']=='functional':print(r)
