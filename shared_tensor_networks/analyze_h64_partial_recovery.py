"""Post-hoc descriptive recovery distributions from the fully audited h64 panel."""
import hashlib,json
from pathlib import Path
import numpy as np
root=Path('results/task_coverage_h64/evaluation');rows=json.loads((root/'results.json').read_text());audit=json.loads((root/'audit.json').read_text());assert audit['status']=='passed' and audit['fits']==60
checked={r['name']:r for r in audit['records']}
for r in rows:assert checked[r['name']]['input_record_sha256']==hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
records=[]
for r in rows:
 c=np.asarray(r['component_cosines']);assert len(c)==64 and np.isfinite(c).all()
 records.append(dict(name=r['name'],teacher=r['teacher'],n=r['n'],tasks=r['tasks'],initialization=r['initialization'],above_99=int(sum(c>.99)),above_95=int(sum(c>.95)),minimum_cosine=float(min(c)),mean_cosine=float(np.mean(c)),transfer=float(np.mean(r['transfer']['learned_8192']['errors']))))
conditions=[]
for n in [8192,32768]:
 for tasks in [520,2080]:
  subset=[r for r in records if r['n']==n and r['tasks']==tasks and r['initialization']!='teacher'];assert len(subset)==10
  teachermeans=[]
  for teacher in sorted({r['teacher'] for r in subset}):
   sub=[r for r in subset if r['teacher']==teacher];assert len(sub)==2
   teachermeans.append(dict(teacher=teacher,mean_above99=float(np.mean([r['above_99'] for r in sub])),mean_above95=float(np.mean([r['above_95'] for r in sub])),mean_transfer=float(np.mean([r['transfer'] for r in sub]))))
  conditions.append(dict(n=n,tasks=tasks,teacher_means=teachermeans,median_teacher_mean_above99=float(np.median([r['mean_above99'] for r in teachermeans])),sorted_run_counts_above99=sorted(r['above_99'] for r in subset),sorted_run_counts_above95=sorted(r['above_95'] for r in subset),all64=sum(r['above_99']==64 for r in subset)))
result=dict(scope='Post-hoc descriptive reanalysis; thresholds .99 (original criterion) and .95 (sensitivity), not new independent replication or causal attribution',conditions=conditions,records=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256={str(root/n):hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['results.json','audit.json']})
out=Path('results/h64_partial_recovery');out.mkdir(exist_ok=False);(out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for c in conditions:print(json.dumps(c),flush=True)
