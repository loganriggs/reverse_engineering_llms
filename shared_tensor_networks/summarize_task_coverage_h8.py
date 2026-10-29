import hashlib,json
from pathlib import Path
import numpy as np
root=Path('results/task_coverage_h8');rows=json.loads((root/'panel/results.json').read_text());audit=json.loads((root/'audit.json').read_text());assert audit['status']=='passed' and audit['fits']==60
summary=[];table=[];paired=[]
for n in [512,1536]:
 for t in [12,36]:
  for init in ['random','teacher']:
   rr=[r for r in rows if r['n']==n and r['tasks']==t and (r['initialization']=='teacher')==(init=='teacher')]
   groups=[dict(teacher=s,transfer=float(np.mean([np.mean(r['transfer']['learned_128']) for r in rr if r['teacher']==s])),source=float(np.mean([np.mean(r['population_mse']) for r in rr if r['teacher']==s]))) for s in range(239101,239106)]
   median=float(np.median([g['transfer'] for g in groups]));recovery=sum(min(r['component_cosines'])>.99 for r in rr)
   summary.append(dict(n=n,tasks=t,labels=n*t,initialization=init,median_transfer=median,recovered=recovery,runs=len(rr),teachers=groups))
   table.append(f'| {n} | {t} | {n*t} | {init} | {median:.6g} | {recovery}/{len(rr)} |')
for seed in range(239101,239106):
 vals={f'n{n}_t{t}':next(g['transfer'] for s in summary if s['n']==n and s['tasks']==t and s['initialization']=='random' for g in s['teachers'] if g['teacher']==seed) for n in [512,1536] for t in [12,36]}
 paired.append(dict(teacher=seed,**vals))
pairtable='\n'.join(f"| {r['teacher']} | {r['n512_t12']:.6g} | {r['n512_t36']:.6g} | {r['n1536_t12']:.6g} | {r['n1536_t36']:.6g} |" for r in paired)
(root/'summary.json').write_text(json.dumps(dict(conditions=summary,paired_teachers=paired),indent=2)+'\n')
sources=['run_task_coverage_h8.py','batched_coverage.py','check_batched_coverage.py','audit_task_coverage_h8.py','TASK_COVERAGE_H8_PLAN.md','quartic_cross_gram.py']
(root/'provenance.json').write_text(json.dumps({name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources},indent=2)+'\n')
text='''# Task coverage versus example count at d16/h8

60completed GPU fits on five fresh shared teachers. Source tasks12/36crossed with inputs512/1536; two random starts and one privileged teacher start each. All train3000full-batch Adam steps atlr.003with per-model clip10, float32/TF32off. No restarts or held-out selection. Exact Gaussian float64 evaluation and four whole held-out tasks with32/128noisy adaptation labels. See TASK_COVERAGE_H8_PLAN.md.

## Primary128label transfer

MSE targets have unit Gaussian second moment. Random medians aggregate both starts within each teacher first; component threshold requires all eight matched absolute cosines>.99. Teacher starts are diagnostics, not a practical learning method.

| Inputs | Tasks | Scalar labels | Initialization | Median teacher transfer MSE | All8components recovered |
|---|---|---:|---|---:|---:|
'''+ '\n'.join(table)+'''

## Every random-start teacher aggregate

| Teacher |512×12|512×36|1536×12|1536×36|
|---|---:|---:|---:|---:|
'''+pairtable+'''

At each fixed input count,36tasks beats12tasks in4/5teacher blocks. This adds labels and readout parameters; it is not a pure diversity intervention. At the equal18432label budget,36tasks×512examples beats12tasks×1536examples in only2/5teachers. The lower median(.00472versus.0247) does not establish a consistent paired advantage. All individual outcomes are retained rather than reporting the favorable median alone.

Increasing input count improves transfer in3/5teachers at either task count. It need not improve finite-budget optimization monotonically. All20teacher-start runs pass the feature-recovery threshold, while random recovery varies5–8/10. The teacher-start gap is evidence of initialization dependence under this recipe, not a proof each poor random fit is a strict local minimum.

Both12and36tasks can recover all eight features in individual random runs. The full36product-span condition is sufficient for the exact theorem, not necessary for every target. Task coverage, input coverage, extra readout parameters, noise averaging and optimizer dynamics are not fully disentangled by this grid.

## Implementation and audit

GPU was idle before launch (RTX5080, no compute applications). Models were batched independently with separate parameter/moment slots. Masked loss is normalized separately by each model's own scalar-label count; gradients are clipped per model. A five-step float64 check matches separate Adam optimizers to1e-12tolerance. CPU/CUDAfloat32 forward and gradient relative errors are1.13e-7and1.72e-7. Inactive readout rows remain exactly unchanged and are excluded from reported active counts (688/1552parameters); equal allocated tensor shapes are not claimed as equal active model sizes.

All60source fits and240learned/oracle adaptation fits independently replay. Literal bilinear forwards verify training losses and target/noise data; SVD verifies adaptation pseudoinverses, including the32label underdetermined case. Gaussian quadratic coefficient inner products independently verify component cosines against the producer's matrix-trace metric. Exact Wick cross-Grams evaluate source/transfer errors; that engine has separate quadrature validation. Dataset/checkpoint hashes, all15logged200-step loss/gradient records per teacher batch, and exact initial pairing across conditions pass. No nonfinite run was dropped.

Teacher and random initial task RMS use exact distributional normalization, an explicit simulation privilege. Teacher factors/readout ranks are checked. Primary128label adaptation has more observations than36features; secondary32label minimum-norm fits depend on the chosen feature coordinates. No generic sample-complexity bound follows.

Runtime is recorded per12-model GPU batch; summing the duplicated per-case batch time would overcount compute. This is not a timing benchmark or a comparison with earlier CPU/TRF work. The previous larger scaling study also differs in input distribution, initialization/calibration, noise, task count and training duration, so this experiment cannot attribute its better recovery to task count alone.

## Decision

Close this frozen grid without tuning the losing cases. It establishes a tractable larger feature-recovery setting and a conditional task-diversity signal, but not a consistent equal-label-budget advantage. Next scaling should preserve the crossed design, active-parameter accounting and whole-task holdouts at a larger width; it should not treat the low median as a confirmed universal benefit. Quantitative robust-identification bounds remain a separate theoretical question. The overall research goal remains active.
'''
Path('TASK_COVERAGE_H8_RESULTS.md').write_text(text)
print('Summary and paired teacher table written.')
