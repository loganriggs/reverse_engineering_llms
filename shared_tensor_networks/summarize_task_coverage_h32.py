import hashlib,json
from pathlib import Path
import numpy as np

root=Path('results/task_coverage_h32');out=root/'panel'
rows=json.loads((out/'results.json').read_text());audit=json.loads((root/'audit.json').read_text())
assert len(rows)==60 and audit['status']=='passed' and audit['fits']==60
seeds=list(range(251101,251106));conditions=[];witnesses=[]
def teacher_means(n,t,kind,key):
 selected=[r for r in rows if r['n']==n and r['tasks']==t and (r['initialization']=='teacher')==(kind=='teacher')]
 def value(r):return np.mean(r['transfer']['learned_2048']) if key=='transfer' else np.mean(r['population_mse'])
 return np.array([np.mean([value(r) for r in selected if r['teacher']==seed]) for seed in seeds])
lines=['# d64/h32: crossed task and input coverage','',
'All 60 fits and 240 adaptation fits are complete and audited. Five fresh teachers, two random starts plus a privileged teacher start per condition, 3,000 full-batch Adam steps. Source tasks132/528 cross inputs2048/8192. Primary adaptation uses2048labels for528hidden-product features;512label adaptation is retained as an underdetermined diagnostic. See TASK_COVERAGE_H32_PLAN.md.','',
'Errors below are medians across five teacher means, averaging both random starts and four held-out tasks within teacher. Teacher starts remain separate. Unit Gaussian target second moment makes MSE1 the zero-predictor reference. Component recovery requires all32matched quadratic features to exceed absolute Gaussian cosine.99.','',
'| Inputs | Tasks | Initialization | Source MSE | Transfer MSE | All32components |',
'|---:|---:|---|---:|---:|---:|']
for n in [2048,8192]:
 for t in [132,528]:
  for kind in ['random','teacher']:
   rr=[r for r in rows if r['n']==n and r['tasks']==t and (r['initialization']=='teacher')==(kind=='teacher')]
   trans=teacher_means(n,t,kind,'transfer');source=teacher_means(n,t,kind,'source')
   rec=sum(min(r['component_cosines'])>.99 for r in rr)
   condition=dict(n=n,tasks=t,initialization=kind,median_transfer=float(np.median(trans)),median_source=float(np.median(source)),recovery=rec,runs=len(rr),teacher_transfer=trans.tolist())
   conditions.append(condition);lines.append(f'| {n} | {t} | {kind} | {condition["median_source"]:.6g} | {condition["median_transfer"]:.6g} | {rec}/{len(rr)} |')
for r in rows:
 if r['initialization']=='teacher':continue
 teacher=next(a for a in rows if a['teacher']==r['teacher'] and a['n']==r['n'] and a['tasks']==r['tasks'] and a['initialization']=='teacher')
 gap=r['train_mse']-teacher['train_mse'];witnesses.append(dict(candidate=r['name'],teacher_start=teacher['name'],empirical_gap=gap,lower_loss=gap>1e-8))
moreinputs=teacher_means(8192,132,'random','transfer');moretasks=teacher_means(2048,528,'random','transfer')
equalwins=int(np.sum(moreinputs<moretasks));fixed={n:int(np.sum(teacher_means(n,528,'random','transfer')<teacher_means(n,132,'random','transfer'))) for n in [2048,8192]}
lines+=['','## Equal-label and fixed-input comparisons','',
f'At equal1,081,344scalar labels,132tasks×8192inputs beats528tasks×2048inputs on {equalwins}/5teacher blocks. Their median transfer errors are {np.median(moreinputs):.6g} and {np.median(moretasks):.6g}, respectively.',
'',f'At fixed input counts,528tasks beats132tasks on {fixed[2048]}/5teachers with2048inputs and {fixed[8192]}/5with8192inputs. This supplies four times as many labels and more readout parameters; it does not isolate task diversity from resources.',
'','## Optimization witnesses','',
f'{sum(w["lower_loss"] for w in witnesses)}/40random fits have empirical loss more than1e-8above their same-condition trained teacher-start checkpoint. Saved checkpoints and independently replayed losses give explicit numerical lower-loss witnesses. They rule out global empirical optimality of those candidate endpoints, but do not certify strict local minima or explain all population error. Teacher initialization is privileged, not a practical recovery algorithm.',
'','## Resource use and validation','']
resources=[json.loads((out/f'{seed}_resources.json').read_text()) for seed in seeds]
lines += [f'Training each12-model teacher batch took {min(r["seconds"] for r in resources):.2f}–{max(r["seconds"] for r in resources):.2f}seconds, excluding CPU dataset construction and exact evaluation. Maximum allocated GPU memory {max(r["peak_allocated_bytes"] for r in resources)/2**30:.3f}GiB; maximum reserved {max(r["peak_reserved_bytes"] for r in resources)/2**30:.3f}GiB. Do not sum the repeated per-case batch times. GPU was idle before launch; no other compute process was observed then.',
'',
'Active parameter counts are73,792/282,880. Inactive readout slots remain byte-identical to initialization. Independent checks replay literal target/noise generation, exact Wick source/transfer errors, SVD adaptation, quadratic-coefficient Gaussian feature matching, all initial pairing and logged curve coverage. Source datasets are stored in four chunks per teacher plus teacher/adaptation metadata. Training is CUDAfloat32 withTF32off; population analysis is CPUfloat64. No nonfinite fits are silently removed.',
'','## Scope and next decision','',
'This extends width/input scaling within a fixed depth-two polynomial family, not depth3–5 recovery or unprivileged circuit discovery. The task/input/width/seed and adaptation counts differ from preceding widths; cross-panel differences cannot be assigned to one factor. Close this fixed grid without optimizer rescue. Use the observed data-coverage and optimization limits to choose the next scaling step; keep the broad research goal active until its remaining requirements are audited.']
Path('TASK_COVERAGE_H32_RESULTS.md').write_text('\n'.join(lines)+'\n')
(root/'summary.json').write_text(json.dumps(dict(conditions=conditions,equal_budget_more_inputs_wins=equalwins,fixed_input_more_tasks_wins=fixed,lower_loss_witnesses=witnesses,resources=resources),indent=2)+'\n')
sources=['TASK_COVERAGE_H32_PLAN.md','run_task_coverage_h32.py','audit_task_coverage_h32.py','summarize_task_coverage_h32.py','coverage_h32_data.py','batched_coverage.py','quartic_cross_gram.py']
(root/'provenance.json').write_text(json.dumps({s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},indent=2)+'\n');print('\n'.join(lines))
