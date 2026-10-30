"""Test resume stability and rejection of altered scientific inputs/results."""
import hashlib,json,shutil,subprocess,sys,tempfile
from pathlib import Path
source=Path('results/h128_runner_smoke_audit/fixture')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(script,root):return subprocess.run([sys.executable,script,'--root',str(root)],capture_output=True,text=True)
before={str(p):sha(p) for p in (source/'evaluation').rglob('*') if p.is_file() and p.name!='worker.lock'}
for script in ['evaluate_task_coverage_h128.py','audit_task_coverage_h128.py']:
 r=run(script,source);assert r.returncode==0,r.stderr
assert before=={str(p):sha(p) for p in (source/'evaluation').rglob('*') if p.is_file() and p.name!='worker.lock'}
checks=[]
with tempfile.TemporaryDirectory() as tmp:
 for mode in ['metric','checkpoint','recipe']:
  root=Path(tmp)/mode;shutil.copytree(source,root);out=root/'evaluation'
  if mode=='metric':
   rows=json.loads((out/'results.json').read_text());rows[0]['source_mse'][0]+=1;name=rows[0]['name'];(out/f'{name}.json').write_text(json.dumps(rows[0]));(out/'results.json').write_text(json.dumps(rows));shutil.rmtree(out/'audit_cases')
  elif mode=='checkpoint':
   case=root/'266101'/'n128_t1'/'267001'/'checkpoint';meta=json.loads((case/'checkpoint.json').read_text());p=case/meta['chunks'][0]['file'];b=bytearray(p.read_bytes());b[-1]^=1;p.write_bytes(b)
  else:
   p=root/'266101'/'source_recipe.json';r=json.loads(p.read_text());r['seed']+=1;p.write_text(json.dumps(r))
  result=run('audit_task_coverage_h128.py',root);assert result.returncode!=0,mode
  checks.append(dict(alteration=mode,rejected=True,exit_code=result.returncode))
out=Path('results/h128_analysis_guard_check');out.mkdir(exist_ok=False)
result=dict(resume_byte_unchanged=True,checks=checks,smoke_audit_sha256=sha(source/'evaluation'/'audit.json'),sources={n:sha(Path(n)) for n in ['check_h128_analysis_guards.py','evaluate_task_coverage_h128.py','audit_task_coverage_h128.py','h128_analysis_io.py']})
(out/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
