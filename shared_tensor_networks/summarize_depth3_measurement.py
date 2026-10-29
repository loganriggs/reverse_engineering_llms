import json
from pathlib import Path
import numpy as np
root=Path('results/depth3_measurement');rows=json.loads((root/'results.json').read_text());audit=json.loads((root/'audit.json').read_text());assert audit['status']=='passed'
old=[r for r in json.loads(Path('results/depth3_hidden_transfer/results.json').read_text()) if r['d']==32 and r['teacher']=='independent']
summary=[];lines=[];paired=[]
for model in ['shared','separate','shared_budget','oracle_source']:
 rr=[r for r in rows if r['model']==model];oo=[r for r in old if r['model']==model]
 prior=np.median([np.mean(np.array(r['relative_mse'])[[1,3]],axis=1) for r in oo],axis=0)
 current=np.median([np.mean(r['relative_mse'],axis=2) for r in rr],axis=0)
 summary.append(dict(model=model,prior_medians=prior.tolist(),current_medians=current.tolist()))
 lines.append(f"| {model} | {prior[0]:.5g} / {prior[1]:.5g} | {current[0,0]:.5g} / {current[0,1]:.5g} | {current[1,0]:.5g} / {current[1,1]:.5g} | {current[2,0]:.5g} / {current[2,1]:.5g} |")
for row in rows:
 z=np.load(root/f"{row['key']}_{row['model']}.npz");b=z['batches'].reshape(8,48);mean=b.mean(axis=0);denom=mean[24:28]
 delta=(mean[4:8]-mean[:4])/denom
 residual=(b[:,4:8]-b[:,:4]-delta*b[:,24:28])/denom
 se=residual.mean(axis=1).std(ddof=1)/np.sqrt(8)
 panelmean=z['batches'].mean(axis=1);panelratio=panelmean[:,:24]/panelmean[:,24:]
 paired.append(dict(key=row['key'],model=row['model'],mean_hidden_minus_source=float(delta.mean()),paired_standard_error=float(se),max_relative_panel_difference=float(np.max(abs(panelratio[0]-panelratio[1])/np.maximum(abs((panelratio[0]+panelratio[1])/2),1e-30)))))
(root/'summary.json').write_text(json.dumps(dict(models=summary,paired=paired),indent=2)+'\n')
old_pass=int(sum(np.sum(np.array(r['precision_pass'])[[1,3]]) for r in old));new_pass=int(sum(np.sum(np.array(r['precision_pass'])[0]) for r in rows));all_pass=int(sum(np.sum(r['precision_pass']) for r in rows))
text='''# Depth-three measurement and calibration replication

Bounded diagnostic replication on the weakest prior regime: d32independent teachers, three replicates, four model/oracle types. Original128label readouts remain fixed; two fresh source-only calibration draws add48readouts. Evaluation uses two independently optimized task-specific proposals, four8192point batches each, all six predictors measured on common points. Same total65536evaluation samples/model as the prior study, but substantially more proposal-optimization work and256rather than32axes per proposal. Not an equal-compute variance comparison.

## Separating measurement from calibration

Each table entry is source-output / hidden-feature relative MSE. Medians over three teacher-level means of four held-out tasks. Zero prediction scores1. “Original norms, new evaluation” is exactly the SAME predictor as the preceding column; only its estimator changed. The other columns change calibration and refit ridge from the same adaptation data.

| Model | Prior evaluation | Original norms, new evaluation | Calibration repeat1 | Calibration repeat2 |
|---|---|---|---|---|
'''+ '\n'.join(lines)+f'''

The broad poor-transfer conclusion survives. Calibration changes some hidden errors, particularly the narrow shared model, but none of these median hidden errors becomes competitive with zero prediction. This is a selected diagnostic regime, not an independent confirmation cohort. The independent teacher-source oracle contains only source-task trunks and is not a held-out-feature oracle.

## Precision improved, not guaranteed

For original fixed predictors, {new_pass}/96task/error estimates pass the numerator/target batchCV<10%gate, versus{old_pass}/96previously. Across all three calibrations,{all_pass}/288pass. Both proposals retain20%uniform support. Every failed estimate remains in the table. These are descriptive batch-variation diagnostics, not rigorous confidence intervals or proof all rare events were sampled. Fitting separate axes per target/error makes precision better here, but does not certify an optimal importance sampler.

The saved summary also gives paired hidden-minus-source mean differences and standard errors using common samples, plus discrepancy between the two proposal panels. With only eight batches these remain diagnostic, not a formal multiple-comparison analysis. No favorable calibration or proposal was selected by test score.

## Calibration has an exact estimator interpretation

CALIBRATION_AS_REGULARIZATION.md derives the equivalent raw-coordinate ridge penalty lambda CᵀD²C. For two normalization draws, the coefficient change satisfies

H1(C1-C0) = -lambda(D1²-D0²)C0,

with H1=FᵀF/N+lambda D1². All48finite-change identities pass. Thus calibration affects the effective regularizer while raw feature functions remain identical. It is distinct from evaluation noise. This does not imply a monotonic generalization effect.

## Audit and decision

Independent NumPy replay verifies all96evaluation batches,24fresh calibration batches per teacher/model set in total96 (two draws×four batches×12cases),72SVDreadout solutions (24original,48new), and pooled ratios/SEs/precision flags. Original norms and readouts are byte-identical to the preceding study. Checkpoint/prior-fit hashes match. The evaluation proposals were fitted only after the six readouts were frozen; no evaluation data changes any fitted predictor.

Close this bounded replication. The remaining16/96imprecise original-predictor estimates stay qualified, rather than triggering an endless sampling escalation. Together with the exact depth2diagnosis and stable d16depth3cases, the evidence supports a limited conclusion: exposing a larger hidden dictionary alone does not fix these trained models' transfer, and measurements must distinguish representation, finite-label estimation, calibration and evaluation. No universal sharing advantage or impossibility theorem is claimed.

Next consolidate these findings into the overall research synthesis and completion audit, preserving the unresolved precision and theoretical scope limits.
'''
# Keep the audit count sentence unambiguous.
text=text.replace('24fresh calibration batches per teacher/model set in total96 (two draws×four batches×12cases)','96fresh calibration batches (two draws×four batches×12cases)')
Path('DEPTH3_MEASUREMENT_RESULTS.md').write_text(text)
print('Precision',old_pass,'->',new_pass,'of96; paired differences',[(r['mean_hidden_minus_source'],r['paired_standard_error']) for r in paired])
