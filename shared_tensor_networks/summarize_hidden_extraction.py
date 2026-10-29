"""Fixed extraction rankings, all teacher blocks and conditional adaptation risk."""
import json,statistics
from pathlib import Path
from run_svhn_standard import digest,write
root=Path(__file__).parent;out=root/'results/hidden_extraction';audit=json.loads((out/'audit.json').read_text());assert audit['status']=='passed' and len(audit['records'])==50
rows=[]
for a in audit['records']:
 p=out/(a['name']+'.json');assert digest(p)==a['result_sha256'] and digest(p.with_suffix('.npz'))==a['checkpoint_sha256'];rows.append(json.loads(p.read_text()))
risk=json.loads((out/'conditional_adaptation_risk.json').read_text());aggregates=[]
for method in ['shared','separate_width','released','teacher']:
 for extraction in ['outputs','hidden_union','quadratic_consensus','quartic_pca']:
  for seed in range(225001,225006):
   a=[r for r in rows if r['method']==method and r['teacher_seed']==seed]
   def error(r,labels):return statistics.mean(next(m['mse'] for m in r['metrics'] if m['extraction']==extraction and m['labels']==labels))
   ra=[r for r in risk['rows'] if r['method']==method and r['teacher_seed']==seed and r['extraction']==extraction and r['labels']==128]
   aggregates.append(dict(method=method,extraction=extraction,seed=seed,transfer=statistics.mean(error(r,128) for r in a),floor=statistics.mean(error(r,'population') for r in a),bias=statistics.mean(statistics.mean(r['conditional_bias_mse']) for r in ra),noise=statistics.mean(r['conditional_noise_variance'] for r in ra)))
lines=['# Hidden information versus usable transfer features','','No source retraining. Fixed extraction on50stored models:15shared,15random-start independenth4,15released-from-shared,5teacher-start independent. Plan HIDDEN_EXTRACTION_PLAN.md fixed ranks and solvers before evaluation. Every original model/failure is retained. Four whole held-out tasks use the existing32/128noisy adaptation labels.','',
'Extraction choices: ten source outputs; within-head hidden-product union (10shared or100independent features, no cross-head products); leading four normalized quadratic function directions followed by their ten products; or leading ten quartic union directions. SVDs are uncentered in the known Gaussian function metric. They use no held-out labels or teacher directions. The rank-four choice is the declared shared-architecture budget, not a rank discovered from data. SVD quadratics need not individually be bilinear: this tests a subspace, not neuron identification.','',
'| Source model | Extraction | Median teacher transfer MSE,128labels | Median population projection floor |','|---|---|---:|---:|']
for method in ['shared','separate_width','released','teacher']:
 for extraction in ['outputs','hidden_union','quadratic_consensus','quartic_pca']:
  a=[r for r in aggregates if r['method']==method and r['extraction']==extraction]
  lines.append(f"| {method} | {extraction} | {statistics.median(r['transfer'] for r in a):.6g} | {statistics.median(r['floor'] for r in a):.6g} |")
lines+=['','Each teacher entry averages all original initialization outcomes and four held-out heads. Teacher-start models have one outcome per teacher, not three independent replications. Every extraction agrees numerically for shared models: their feature spans are the same, providing a useful control.','',
'## Every teacher: outputs versus quadratic consensus','','| Source model | Teacher | Outputs | Four-quadratic consensus |','|---|---|---:|---:|']
for method in ['separate_width','released','teacher']:
 for seed in range(225001,225006):
  a={r['extraction']:r for r in aggregates if r['method']==method and r['seed']==seed}
  lines.append(f"| {method} | {seed} | {a['outputs']['transfer']:.6g} | {a['quadratic_consensus']['transfer']:.6g} |")
lines+=['','## Main qualification to the previous result','',
'Releasing sharing did not erase all useful hidden information. Quadratic consensus reduces released-model median transfer MSE from.03436to.001215without additional source training or teacher-guided feature selection. Teacher-start independent models show the same median recovery. This remains worse than the shared median.000852and does not repair every failed shared initialization. Quartic rank-ten compression performs similarly, so this does not demonstrate a unique benefit of quadratic rather than quartic compression.','',
'Random-start independent models are different: neither fixed compression reliably recovers their useful directions. Their100-feature union has a small population projection floor (.000102median), showing substantial representational coverage, but noisy128-label transfer is poor (.187median). Thus an expressive learned union is not automatically a sample-efficient transfer representation. The union has more adaptation parameters than the ten-feature methods.','',
'## Exact conditional adaptation-risk decomposition','',
'Let F contain feature-function coefficients, G their ambient Gaussian monomial Gram, X the adaptation feature design, and Y0 the noiseless adaptation targets. The fixed minimum-norm estimator with iid label noise E of variance sigma² is B=X⁺(Y0+E). For each task with coefficient vector c, conditional on the features and adaptation inputs,','',
'E[population MSE] = ||F X⁺Y0−c||²_G + sigma² ||G^(1/2) F X⁺||²_F.','',
'The first term includes finite-input approximation bias, not just the population projection floor. The second is pure adaptation-noise variance. This formula also applies to underdetermined designs with the declared pseudoinverse cutoff; it is not an assumption that quartic features themselves are Gaussian.','',
'| Source model | Extraction | Median conditional bias | Median noise contribution |','|---|---|---:|---:|']
for method in ['separate_width','released','teacher']:
 for extraction in ['outputs','hidden_union','quadratic_consensus']:
  a=[r for r in aggregates if r['method']==method and r['extraction']==extraction]
  lines.append(f"| {method} | {extraction} | {statistics.median(r['bias'] for r in a):.6g} | {statistics.median(r['noise'] for r in a):.6g} |")
lines+=['','The independent union has noise contribution about.20, versusabout.0005for ten-feature extraction. This quantitatively explains why the union can represent the target yet transfer poorly with128noisy labels. The conditional expectation is an evaluation diagnostic, not a selection criterion used to tune extraction.','',
'## Verification and scope','',
'All50models/600projection or adaptation fits independently audited: source provenance,400fresh-input evaluation of quadratic/product features, normalization, leading-eigenspace equations, adaptation normal equations plus minimum-norm row-space condition, exact Gaussian errors and teacher-subspace evaluation.400conditional-risk decompositions replay actual noisy fits and agree between whitened andQRvariance formulas within1.39e-11. A naive Gram-trace calculation showed cancellation error up to.000153and was replaced by the stable factored formulas; that diagnostic is retained.','',
'Sharing still improves practical learning in the earlier controls, but the output-span transfer gap overstates how much hidden information was destroyed by untying. Coherent feature extraction can recover much of it after a good shared/teacher start. The random-start extraction failures and the privileged Gaussian metric/rank budget limit broader claims. No native-model featurizer work was restarted.','']
(root/'HIDDEN_EXTRACTION_RESULTS.md').write_text('\n'.join(lines));write(out/'summary.json',dict(status='complete',aggregates=aggregates,source_sha256=digest(Path(__file__))))
print('\n'.join(lines))
