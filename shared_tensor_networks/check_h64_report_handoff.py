"""Synthetic audit-to-report contract; does not certify scientific outcomes."""
import hashlib,json,shutil,subprocess,sys,tempfile
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
source=Path.cwd();rows=[];groups=[]
for seed in range(256101,256106):
 for n in [8192,32768]:
  for tasks in [520,2080]:
   groups.append(dict(teacher=seed,n=n,tasks=tasks,resources=dict(seconds=1.,peak_allocated_bytes=2**30)))
   for initial in ['257001','257002','teacher']:
    value=.001 if initial=='teacher' else .1
    rows.append(dict(name=f'{seed}_n{n}_t{tasks}_{initial}',teacher=seed,n=n,tasks=tasks,initialization=initial,train_mse=value,population_mse=[value]*tasks,component_cosines=[1. if initial=='teacher' else .9]*64,transfer={'learned_8192':{'errors':[value]*4}}))
with tempfile.TemporaryDirectory(prefix='h64_report_handoff_') as temp:
 root=Path(temp)
 for name in ['summarize_task_coverage_h64.py','aggregate_h64.py']:shutil.copy(source/name,root/name)
 out=root/'results/task_coverage_h64/evaluation';cache=out/'audit_cases';cache.mkdir(parents=True)
 dump(out/'results.json',rows);dump(out/'protocol.json',{'synthetic_fixture':True})
 (root/'fixture_source.py').write_text('# Synthetic source hash fixture.\n')
 provenance=dict(results_sha256=sha(out/'results.json'),evaluation_protocol_sha256=sha(out/'protocol.json'),source_sha256={'fixture_source.py':sha(root/'fixture_source.py')})
 dump(cache/'protocol.json',provenance)
 audit=dict(status='passed',fits=60,records=[dict(name=r['name'],input_record_sha256=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()) for r in rows])
 dump(out/'audit.json',audit);dump(out.parent/'training_progress.json',dict(groups=groups))
 command=[sys.executable,'summarize_task_coverage_h64.py']
 result=subprocess.run(command,cwd=root,capture_output=True,text=True);assert result.returncode==0,result.stderr
 summary=json.loads((out/'summary.json').read_text());assert len(summary['resources'])==20 and sum(r['seconds'] for r in summary['resources'])==20
 assert len(summary['conditions'])==8 and sum(r['recovery'] for r in summary['conditions'])==20
 assert '120 learned transfer fits and 10 shared oracle fits' in (root/'TASK_COVERAGE_H64_RESULTS.md').read_text()
 rejected=[]
 def fails(label):
  result=subprocess.run(command,cwd=root,capture_output=True,text=True);assert result.returncode!=0,label;rejected.append(label)
 audit['status']='pending';dump(out/'audit.json',audit);fails('unfinished_audit');audit['status']='passed';dump(out/'audit.json',audit)
 original=(out/'results.json').read_bytes();rows[0]['train_mse']+=1;dump(out/'results.json',rows);fails('changed_result_hash')
 provenance['results_sha256']=sha(out/'results.json');dump(cache/'protocol.json',provenance);fails('changed_result_case_hash')
 (out/'results.json').write_bytes(original);provenance['results_sha256']=sha(out/'results.json');dump(cache/'protocol.json',provenance)
 (root/'fixture_source.py').write_text('# Changed\n');fails('changed_source')
out=source/'results/h64_report_handoff';out.mkdir(exist_ok=True)
dump(out/'checks.json',dict(scope='Synthetic report contract; not scientific audit',conditions=8,resource_groups=20,resource_seconds=20,rejected=rejected,source_sha256={n:sha(source/n) for n in ['check_h64_report_handoff.py','summarize_task_coverage_h64.py','aggregate_h64.py']}))
print('Report handoff passed; four invalid-state checks rejected.')
