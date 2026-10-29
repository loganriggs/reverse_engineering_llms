import hashlib,json
from pathlib import Path
import numpy as np
root=Path('results/task_coverage_h16');out=root/'panel';rows=json.loads((out/'results.json').read_text());a=json.loads((root/'audit.json').read_text());assert a['status']=='passed' and a['fits']==60
summary=[];table=[];witnesses=[];pairs=[]
for n in [512,2048]:
 for t in [34,136]:
  for kind in ['random','teacher']:
   rr=[r for r in rows if r['n']==n and r['tasks']==t and (r['initialization']=='teacher')==(kind=='teacher')]
   groups=[dict(teacher=s,transfer=float(np.mean([np.mean(r['transfer']['learned_512']) for r in rr if r['teacher']==s])),source=float(np.mean([np.mean(r['population_mse']) for r in rr if r['teacher']==s]))) for s in range(242101,242106)]
   recovery=sum(min(r['component_cosines'])>.99 for r in rr);med=float(np.median([g['transfer'] for g in groups]));summary.append(dict(n=n,tasks=t,labels=n*t,initialization=kind,median_transfer=med,recovery=recovery,runs=len(rr),teachers=groups))
   table.append(f'| {n} | {t} | {n*t} | {kind} | {med:.6g} | {recovery}/{len(rr)} |')
for r in rows:
 if r['initialization']=='teacher':continue
 teacher=next(x for x in rows if x['teacher']==r['teacher'] and x['n']==r['n'] and x['tasks']==r['tasks'] and x['initialization']=='teacher')
 gap=r['train_mse']-teacher['train_mse'];witnesses.append(dict(candidate=r['name'],lower_loss_candidate=teacher['name'],empirical_loss_gap=gap,lower_by_more_than_1e_minus8=gap>1e-8))
for seed in range(242101,242106):
 v={f'n{n}_t{t}':next(g['transfer'] for s in summary if s['n']==n and s['tasks']==t and s['initialization']=='random' for g in s['teachers'] if g['teacher']==seed) for n in [512,2048] for t in [34,136]};pairs.append(dict(teacher=seed,**v))
resources=[json.loads((out/f'{s}_resources.json').read_text()) for s in range(242101,242106)]
(root/'summary.json').write_text(json.dumps(dict(conditions=summary,paired_teachers=pairs,lower_loss_witnesses=witnesses,resources=resources),indent=2)+'\n')
pairtable='\n'.join(f"| {r['teacher']} | {r['n512_t34']:.6g} | {r['n512_t136']:.6g} | {r['n2048_t34']:.6g} | {r['n2048_t136']:.6g} |" for r in pairs)
text='''# d32/h16: input coverage beats task count at equal total labels

60GPU fits and240transfer fits complete and audited. Five fresh shared teachers, source tasks34/136crossed with512/2048inputs, two random starts plus privileged teacher start,3000full-batch Adam steps. Same source/adaptation noise SD.05, fixed recipe, no restarts. Primary transfer uses512labels for136hidden-product features;128label transfer is retained as an underdetermined secondary diagnostic.

## Primary512label transfer

Unit Gaussian target second moments; MSE1is zero prediction. Random medians average both initializations within each teacher first. All16component criterion is matched absolute Gaussian cosine>.99for every component; this is stricter than recovering eight components at the preceding width.

| Inputs | Tasks | Scalar labels | Initialization | Median teacher transfer MSE | All16components recovered |
|---|---|---:|---|---:|---:|
'''+ '\n'.join(table)+'''

## Every random-start teacher aggregate

| Teacher |512×34|512×136|2048×34|2048×136|
|---|---:|---:|---:|---:|
'''+pairtable+'''

At equal69632scalar labels,34tasks with2048inputs beats136tasks with512inputs in ALL5teacher blocks. Medians are.0322versus1.536. In this regime, distributing a fixed budget over more input examples is much more useful than allocating it to the complete task-product span with only512inputs.

At fixed inputs,136tasks improves transfer in4/5teachers for512inputs and5/5for2048inputs. But this also supplies four times as many labels and more readout parameters. Full task span therefore helps conditionally, while being insufficient to guarantee useful random-start learning at low input coverage.

Both512input random conditions recover zero of ten full feature sets and have median transfer error worse than zero prediction. All20teacher-start runs recover all16features and transfer well. With2048inputs,136tasks recovers6/10random starts versus2/10for34tasks. This establishes a larger tractable regime, not robust recovery from every start.

## Explicit optimization witnesses

Every low-input random fit has a higher empirical training loss than its corresponding trained teacher-start checkpoint on EXACTLY the same inputs, labels and architecture. The counts across(512,34),(512,136),(2048,34),(2048,136)are10/10,10/10,9/10,5/10using a numerical margin1e-8. The60fit audit independently replays those losses. The checkpoint identities and margins are saved in summary.json.

A feasible lower-loss parameter vector rules out the higher-loss point as a global empirical minimizer. These are explicit numerical witnesses, not interval-certified arithmetic or proof of strict local minima. They do not show optimization explains all population error; poor fits can also generalize badly. Teacher starts are privileged and are not proposed as a practical estimator.

## Controls, resources and limits

Active parameter counts are5648/19520. Batched tensors allocate136readout rows for every case, but unused heads have zero gradients and stay byte-identical to initialization. They are excluded from active counts. Exact float32 initial functions match across input-count arms and task prefixes. The independently verified batched Adam implementation and CPU/CUDA gradient controls from the previous panel are unchanged.

All60fits complete3000steps with no nonfinite losses/gradients dropped. Independent literal forward/data checks, SVD transfer fits, Gaussian quadratic component metrics, exact Wick source/transfer errors, curve coverage, inactive-head and initialization pairing checks pass. Population evaluation is CPUfloat64; training is CUDAfloat32withTF32off. Known Gaussian initial/teacher RMS normalization remains an explicit simulation privilege.

Training each12model teacher batch took approximately6.2–6.6seconds on RTX5080, excluding CPU target construction/final exact analysis. Peak torch-allocated memory was about195MB and peak reserved memory about229MB; these are allocator measurements, not total device memory. The GPU had no external compute jobs before launch. Per-case records repeat their teacher-batch duration and must not be summed as60independent durations.

The source task/input counts, width, seeds and primary adaptation budget differ from h8. This is controlled scaling within this panel, not proof that one variable alone explains differences across panels. More task labels, head parameters, task geometry and optimizer dynamics remain coupled. No generic sample-complexity threshold follows, and the full-span theorem remains sufficient rather than necessary.

## Decision

Close this frozen grid. It supports increasing input coverage along with task diversity when scaling width; task count alone does not repair optimization. The explicit teacher-start solutions demonstrate representability and accessible good solutions under privileged initialization. Further scaling should preserve input-budget controls and memory accounting, not interpret the earlier low median task-diversity result as a universal prescription. The broader proposed scaling envelope remains unfinished.
'''
Path('TASK_COVERAGE_H16_RESULTS.md').write_text(text)
sources=['TASK_COVERAGE_H16_PLAN.md','run_task_coverage_h16.py','audit_task_coverage_h16.py','batched_coverage.py','quartic_cross_gram.py']
(root/'provenance.json').write_text(json.dumps({s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},indent=2)+'\n');print('Summary written; lower-loss witnesses',sum(r['lower_by_more_than_1e_minus8'] for r in witnesses),'/40')
