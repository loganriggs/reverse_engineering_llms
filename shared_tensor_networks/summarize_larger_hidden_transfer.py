import json
from pathlib import Path
import numpy as np
root=Path('results/larger_hidden_transfer');rows=json.loads((root/'results.json').read_text());floor=json.loads((root/'population_floors.json').read_text());audit=json.loads((root/'audit.json').read_text());assert audit['status']=='passed' and audit['fits']==192
summary=[];table=[]
for d,gi in [(16,0),(32,2)]:
 for teacher in ['shared','independent']:
  for model in ['shared','separate','shared_budget','oracle_source']:
   cell=dict(d=d,teacher=teacher,model=model)
   for representation in ['source','hidden']:
    rr=[r for r in rows if r['d']==d and r['teacher']==teacher and r['model']==model and r['representation']==representation]
    ff=[r for r in floor if r['key'].startswith(f'g{gi}_{teacher}_') and r['model']==model and r['representation']==representation]
    cell[representation]=dict(features=rr[0]['features'],transfer32=float(np.median([np.mean(r['relative_mse']) for r in rr if r['n']==32])),transfer128=float(np.median([np.mean(r['relative_mse']) for r in rr if r['n']==128])),floor=float(np.median([np.mean(r['floor_checks'][1]['relative_floor']) for r in ff])))
   summary.append(cell)
   table.append(f"| {d} | {teacher} | {model} | {cell['hidden']['features']} | {cell['source']['transfer128']:.5g} | {cell['hidden']['transfer128']:.5g} | {cell['source']['floor']:.5g} | {cell['hidden']['floor']:.5g} |")
# Nested source output span must not improve on hidden population projection.
for f in floor:
 if f['representation']=='source':
  hidden=next(r for r in floor if r['key']==f['key'] and r['model']==f['model'] and r['representation']=='hidden')
  assert np.all(np.array(hidden['floor_checks'][1]['relative_floor'])<=np.array(f['floor_checks'][1]['relative_floor'])+1e-7)
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
text='''# Larger frozen hidden dictionaries: information loss is not the whole scaling failure

192adaptation fits complete:12depth2teacher groups (d16/h8,d32/h16; shared/independent;3replicates), three frozen learned models plus teacher-source oracle, source-output/hidden representations, and32/128labels. No retraining, new labels, added adaptation noise, or penalty tuning. All36learned models are included. This closes the depth2portion of the previously deferred scaling diagnosis; depth3is not included.

## Results at128adaptation labels per held-out task

Errors are relative population MSE (1=zero predictor), exact on the radius-sqrt(d) sphere. Entries are medians over three teachers after averaging their four held-out tasks. Source features always have dimension12. Floor means best population linear readout from that frozen representation, not an attainable finite-label estimator.

| d | Teacher | Model | Hidden dimension | Source transfer | Hidden transfer | Source floor | Hidden floor |
|---|---|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

## Interpretation

The12output bottleneck is real: all hidden dictionaries have population floors no worse than their source-output spans, as inclusion requires. Some contain substantially more relevant information. For d16shared teachers, separate and near-budget dictionaries reduce median floors from roughly.67/.61to.27/.27.

But hidden features do not broadly rescue finite-label transfer under the fixed recipe. At d32, the shared student's median hidden floor is still.873, versus zero for the shared teacher's true dictionary. Its128label error reaches21.85despite the lower floor. These are two distinct limitations: incomplete learned representation and poor finite-sample readout estimation. There is NO newly added label noise in these archived adaptation targets, so the latter cannot be attributed to label-noise amplification alone. Misspecification, sampled design, ridge bias and representation dimension all matter.

The true shared hidden dictionary provides a positive control: d16/h8median128label error is about1e-5, versus.565using its12source outputs; d32/h16it is.0694versus.959. At h16there are136products but only128adaptation examples. Even true features therefore need not yield zero finite-label error. The independent-teacher oracle includes ONLY the12source trunks; it has no access to the four independent held-out trunks and is not a true held-out-feature oracle.

## Protocol and audit

Both source and hidden features are normalized by exact sphere RMS, with fixed mean-loss ridge .001. The source comparator is refitted under this convention; original scaling used sampled feature RMS. Exact normalization uses the known simulation distribution, not target labels, but is still a privileged distributional calculation compared with a practical finite-unlabeled-data estimator. Within-head unions add no cross-head products.

Wick cross-Grams integrate degree-eight products; radius-sqrt(d) sphere moments equal Gaussian moments times d^4/[d(d+2)(d+4)(d+6)]. The underlying generic/structured moment engines pass independent Gauss–Hermite fixtures. All48model/oracle dictionaries replay against the actual network's source and hidden forwards. All192readouts agree with independent SVD ridge fits. Population errors are replayed through the checked exact-moment engine. Hashes tie every fit to frozen data/checkpoints. “Exact” here means exact moment identities evaluated in float64, not arbitrary-precision certification.

Post-hoc population-floor diagnostics use normalized Gram eigendecomposition with two fixed numerical cutoffs1e-10and1e-12relative to the largest eigenvalue; both give identical retained-floor results here. PSD tolerances and nested-span floor inequalities pass. These truth-aware diagnostics do not tune the actual adaptation fits.

All72historical depth2source-output readouts also replay using their original sampled normalizers. Their exact errors are saved alongside the prior Monte Carlo estimates; maximum absolute per-task discrepancy is.2304. This is a precision diagnostic, not evidence of changed models. The previous Monte Carlo precision qualifications remain relevant. No historical estimate is silently overwritten.

## Next scope

Do not answer these results with another unconstrained optimizer sweep. The missing task-output span is insufficient to explain the larger scaling failures: direct features are both incompletely learned and difficult to estimate from few examples. Next complete the analogous frozen-representation diagnosis at depth3with explicit evaluation uncertainty, or establish that the available precision cannot support its ranking. Keep population representational floor, finite-label transfer and source prediction separate. No new universal learnability or circuit-recovery claim follows.
'''
Path('LARGER_HIDDEN_TRANSFER_RESULTS.md').write_text(text)
print('Summary and all nested-span checks passed.')
