import json
from pathlib import Path
import numpy as np
root=Path('results/crossed_readout');rows=json.loads((root/'results.json').read_text());audits=json.loads((root/'audit.json').read_text())
old=Path('results/normalized_readout/panel')
for r in json.loads((old/'results.json').read_text()):r.update(condition='unit_ridge',raw_max_norm=1+r['radial_max_error']);rows.append(r)
for r in json.loads((old/'audit.json').read_text()):r['condition']='unit_ridge';audits.append(r)
summary=[];lines=[]
for c in ['raw_zero','raw_ridge','unit_zero','unit_ridge']:
 for m in ['joint','reduced']:
  r=[x for x in rows if x['condition']==c and x['method']==m]
  teacher=[dict(seed=s,transfer=float(np.mean([x['transfer_mse'] for x in r if x['seed']==s])),source=float(np.mean([x['source_mse'] for x in r if x['seed']==s]))) for s in range(232101,232106)]
  recovery=sum(x['all_components_above_099'] for x in audits if x['condition']==c and x['method']==m)
  out=dict(condition=c,method=m,recovered=recovery,median_teacher_transfer=float(np.median([x['transfer'] for x in teacher])),teacher_aggregates=teacher,max_raw_norm=max(x['raw_max_norm'] for x in r),capped=sum(x['status']==0 for x in r))
  summary.append(out);lines.append(f"| {c} | {m} | {recovery}/10 | {out['median_teacher_transfer']:.6g} | {out['max_raw_norm']:.4g} | {out['capped']}/10 |")
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
text='''# Crossed readout dynamics: normalization versus regularization

60newfits plus20archived fits complete the four-cell panel on identical five teachers, two starts each,512source examples, ten source tasks and four held-out tasks with128adaptation labels. Two methods per cell: joint versus exact readout elimination. TRF200evaluation cap, no rescues or new tuning. Teacher-mean transfer averages both starts before taking the median across five teachers. Do not pool this with the earlier d6 pilot.

| Coordinates / penalty | Method | All4features cosine>.99 | Median teacher transfer MSE | Max raw row norm | Hit cap |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(lines)+'''

## Interpretation

Removing ridge does not remove the unit-factor reduced method's recovery deficit: joint9/10 versus reduced5/10 without ridge,9/10 versus6/10 with ridge. Raw/no-ridge methods both recover8/10. Equal counts and median errors do not imply identical trajectories or cases. Thus the normalization/coordinate package materially changes method behavior, and the small ridge term alone cannot explain the reduced failures. This package includes a radial penalty as well as normalized predictions; the experiment does not isolate those ingredients separately.

Raw/ridge is a diagnostic of a known scale escape, not a recommended estimator: every joint run reaches the cap, while reduced raw norms grow as high as527. Raw/no-ridge reduced norms reach757. Algorithm termination does not mean global optimality. No failed or capped case is removed.

The functions available with nonzero factors are unchanged by row normalization when readouts can compensate. At zero ridge, the reduced objective is invariant to nonzero factor scaling, yet its parameter-space trust-region dynamics need not be invariant. Coordinate choice is consequential even after eliminating the readout.

## Verification

All60new checkpoints independently replay direct versus polynomial predictions, exact Gaussian source errors, penalized objectives, QR readout optima (reduced), transfer OLS and exact population errors. All adaptation designs have rank10. Checkpoint and source-data hashes match. Eight crossed objective/Jacobian fixtures agree with independent autograd to maximum relative error9.69e-14. The20archived controls retain their prior independent audit. Unit-factor controls contain no activation RMSNorm.

A separate exact-geometry check in10initial teacher/penalty fixtures establishes a mechanism worth isolating next: see PROFILED_GAUSS_NEWTON_GEOMETRY.md. It is not yet a causal explanation for the final success counts.

## Decision

Close this normalization/ridge sweep. Do not search more penalties or seeds to recover a preferred ranking. The specific next optimizer control is to compare the full reduced-residual Gauss–Newton matrix with the profiled joint-linearization matrix, using the same reduced objective, gauge convention, damping and line search. This isolates the newly derived positive-semidefinite term. Preserve failed starts and held-out transfer; a change in source loss alone is insufficient.
'''
Path('CROSSED_READOUT_RESULTS.md').write_text(text)
