import json
from pathlib import Path
import numpy as np
root=Path('results/depth3_hidden_transfer');rows=json.loads((root/'results.json').read_text());audit=json.loads((root/'audit.json').read_text());assert audit['status']=='passed' and audit['models']==48
summary=[];lines=[]
for d in [16,32]:
 for kind in ['shared','independent']:
  for model in ['shared','separate','shared_budget','oracle_source']:
   rr=[r for r in rows if r['d']==d and r['teacher']==kind and r['model']==model];cell=dict(d=d,teacher=kind,model=model)
   for rep,i in [('source',1),('hidden',3)]:
    cell[rep]=dict(median_teacher_error=float(np.median([np.mean(r['relative_mse'][i]) for r in rr])),precision_passes=int(sum(sum(r['precision_pass'][i]) for r in rr)))
   cell['median_bad_calibration_fraction']=float(np.median([r['calibration_bad_fraction'] for r in rr]));summary.append(cell)
   lines.append(f"| {d} | {kind} | {model} | {cell['source']['median_teacher_error']:.5g} | {cell['hidden']['median_teacher_error']:.5g} | {cell['source']['precision_passes']}/12 | {cell['hidden']['precision_passes']}/12 |")
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
passed=int(sum(np.sum(r['precision_pass']) for r in rows));cal=float(np.median([r['calibration_bad_fraction'] for r in rows]));maxreplay=max(r['max_relative_batch_replay_error'] for r in audit['records'])
text='''# Depth-three frozen hidden transfer: negative pipeline result with precision limits

All48model/oracle evaluations and192adaptation fits complete. Same12teacher groups,36saved learned models,32/128adaptation examples, source12outputs versus within-head hidden-product features. No source retraining or new adaptation noise. Teacher-source oracle for independent teachers contains only source trunks, not the four held-out trunks.

##128label transfer estimates

Relative MSE1means zero prediction. Each error is the median over three teacher replicates after averaging four held-out tasks. Pass counts are individual task estimates passing both numerator/target batchCV<10%; they are not independent teacher replications. No failures are filtered from the medians.

| d | Teacher | Model | Source estimate | Hidden estimate | Source precision | Hidden precision |
|---|---|---|---:|---:|---:|---:|
'''+ '\n'.join(lines)+f'''

## What can be concluded

At d16shared teachers, all128label task estimates pass the descriptive evaluation gate. Learned hidden transfer is worse than source-output transfer under this fixed recipe: median errors about3.65(shared),3.14(separate),2.36(wider shared), versus1.47,1.00,1.16from source outputs. In contrast, the true shared hidden dictionary achieves about.0039versus.625from source outputs. Thus the measured finite-label pipeline has a substantial learned-versus-oracle gap, and simply exposing more hidden coordinates does not repair it.

Do not infer an irreducible representation floor from these errors: no depth3population projection floor was computed. Poor finite-label readout estimation, incomplete features and sampled normalization can all contribute. The exact depth2floor results cannot be transferred to degree8features by analogy.

At d32independent teachers, many or all task estimates fail the precision gate. Those medians are exploratory estimates, not defensible fine rankings. Across both label counts and all models, {passed}/768individual task/error estimates pass. The historical depth2result used exact moments; this stage does not.

## Normalization is another uncertainty

Feature RMS estimates use four independent4096point importance-weighted batches with source-only archived axes. The median fraction of columns exceeding10%batchCV in second moments is {cal:.3f}. This is not a percentage error in the norms and not a failure count of model training. It means sampling variation in normalization is material and can change the effective ridge penalty.

For each saved predictor, the evaluation estimates its actual population error conditional on those fixed norms/readouts. But this does not establish how accurately its norms approximate population RMS, nor how a different calibration draw would change training the readout. The positive oracle result does not erase this limitation.

## Evaluation and audit

Each readout is fixed before held-out target functions are used to fit evaluation axes. The evaluation proposal targets both true energy and residual energy, then uses eight fresh8192point batches with20%uniform defensive mixture. Common points within a model allow paired comparisons. Proposal adaptation sees targets only for evaluating fixed predictions; it does not alter source training, calibration, ridge, adaptation labels or readout coefficients.

Saved batch means permit ratio-of-means relative errors and paired delta-method standard errors. These are descriptive uncertainty estimates with only eight batches, not rigorous coverage guarantees; rare regions can still be missed. Calibration and evaluation seeds are distinct.

Independent NumPy forward code replays all384evaluation batches,192SVD ridge fits and48feature-calibration jobs; maximum relative batch replay discrepancy {maxreplay:.3g}. Checkpoint/data hashes match. Degree-eight homogeneity passes, and teacher-heldout parameter mutations leave oracle source features unchanged. Four independent forward/degree fixtures and two depth2exact-moment fixtures validate feature ordering/evaluation primitives. This supports correctness, not guaranteed Monte Carlo accuracy.

## Decision

The deferred finite-label hidden-transfer diagnosis now covers both feasible depths2/3and all saved larger teacher regimes. Depth2supports exact population statements; depth3supports the above conditional pipeline findings with explicit unresolved precision/calibration uncertainty. Do not treat192completed fits as192precise conclusions.

Next prioritize a bounded calibration/evaluation replication for the weak d32independent regime or explicitly limit scientific claims to the stable settings in the final synthesis. Avoid another optimizer sweep: the remaining uncertainty here is measurement and finite-label representation use. Broader robust identifiability and universal sharing advantages remain unproved.
'''
Path('DEPTH3_HIDDEN_TRANSFER_RESULTS.md').write_text(text.replace('##128','## 128'));print('Summary written; precision',passed,'/768')
