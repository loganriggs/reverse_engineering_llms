"""Check audit gating and smoke-label handling before long-run report handoff."""
import hashlib,json,shutil,subprocess,sys,tempfile
from pathlib import Path
source=Path('results/h128_runner_smoke_audit/fixture');summary=json.loads((source/'evaluation/summary.json').read_text())
assert len(summary['conditions'])==8 and summary['equal_budget']==dict(eligible=False,labels=None,differences=None,more_inputs_wins=None)
assert all(c['runs']==(1 if c['initialization']=='teacher' else 2) for c in summary['conditions'])
progress=json.loads((source/'progress.json').read_text())['completed'];assert len(progress)==12
actual=0.
for r in progress:actual+=json.loads((source/str(r['teacher'])/f'n{r["n"]}_t{r["tasks"]}'/r['initialization']/'trained.json').read_text())['resources']['seconds']
assert actual==summary['training_seconds'];checks=[]
with tempfile.TemporaryDirectory() as tmp:
 for mode in ['pending_audit','changed_result','missing_case']:
  root=Path(tmp)/mode;shutil.copytree(source,root)
  if mode=='pending_audit':p=root/'evaluation/audit.json';v=json.loads(p.read_text());v['status']='pending'
  elif mode=='changed_result':p=root/'evaluation/results.json';v=json.loads(p.read_text());v[0]['train_mse']+=1
  else:p=root/'progress.json';v=json.loads(p.read_text());v['completed'].pop()
  p.write_text(json.dumps(v));r=subprocess.run([sys.executable,'summarize_task_coverage_h128.py','--root',str(root)],capture_output=True,text=True);assert r.returncode!=0,mode
  checks.append(dict(alteration=mode,rejected=True))
out=Path('results/h128_report_handoff');out.mkdir(exist_ok=False)
result=dict(conditions=8,independent_resource_records=12,unequal_smoke_budget_not_mislabeled=True,negative_checks=checks,sources={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['summarize_task_coverage_h128.py','check_h128_report_handoff.py']})
(out/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
