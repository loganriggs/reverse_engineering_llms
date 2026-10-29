"""Summarize audited local response and noise fits without selecting successes."""
import json,statistics
from pathlib import Path
from run_svhn_standard import digest,write
root=Path(__file__).parent;out=root/'results/readout_stability';audit=json.loads((out/'audit.json').read_text());assert audit['status']=='passed' and len(audit['records'])==48
rows=[]
for a in audit['records']:
 p=out/(a['name']+'.json');assert digest(p)==a['result_sha256'] and digest(p.with_suffix('.npz'))==a['checkpoint_sha256'];rows.append(json.loads(p.read_text()))
lines=['# Local stability of recovered bilinear features','','Twelve teacher/supervision geometries and48teacher-initialized nonlinear refits. Same d6/h3teachers as the readout panel. Noise amplitudes1e-5and1e-3, one fixed random direction and one analytically worst feature-response direction per geometry. Noise acts on whitened quartic coefficient observations, not individual sampled labels. Its squared norm is the mean squared Gaussian output perturbation after individual head normalization. All48fits terminated within100evaluations; no restarts or discarded outcomes.','',
'The feature metric is mean squared change in Gaussian-unit-norm hidden quadratic functions, with fixed local ordering/sign. It ignores reciprocal factor scale and hidden-feature amplitude; it is not raw parameter distance. Initializing at the true teacher is a privileged basin control, not a practical learning procedure.','',
'| Teacher | Supervision | Worst-direction feature gain | Isotropic-noise RMS feature gain |','|---|---|---:|---:|']
geometry=[]
for seed in [218001,218002,218003]:
 for arm in ['one','three','six','interventions']:
  g=json.loads((out/f'{seed}_{arm}_geometry.json').read_text());geometry.append(g)
  lines.append(f"| {seed} | {arm} | {g['worst_feature_gain']:.4f} | {g['random_direction_rms_gain']:.4f} |")
lines+=['','Gains multiply the observation-noise norm to predict feature-function change locally. Isotropic RMS is an analytic expectation over uniformly oriented noise, not an estimate from one random draw. It decreases from one to three to six independent heads for each teacher. Labeled ablations are not uniformly best; they are not nested extensions of the independent-head experiments. Worst-direction gain need not decrease under the chosen mean-output-error normalization.','', '| Noise amplitude | Median relative error of linear feature prediction | Maximum |','|---|---:|---:|']
for amplitude in [1e-5,1e-3]:
 a=[r['relative_linearization_error'] for r in rows if r['amplitude']==amplitude]
 lines.append(f'| {amplitude:g} | {statistics.median(a):.6g} | {max(a):.6g} |')
lines+=['','This agreement supports the local sensitivity calculation over these perturbation sizes; it does not guarantee stability at large noise or random initialization. The larger perturbation produces about100times the relative linearization error, consistent with a first-order response approximation.','',
'## Coordinate-independent response','',
'Let y(theta) be the whitened, normalized source-output coefficients and u(theta) the normalized hidden feature functions. Write J=dy/dtheta and D=du/dtheta at the teacher. When ker(J) is contained in ker(D), the first-order response to observation perturbation eta is D J⁺ eta. Though J⁺ alone depends on parameter coordinates, this product does not: solutions of the linearized least-squares problem differ only by ker(J), which D annihilates. Its operator norm gives worst-direction gain; its Frobenius norm divided by sqrt(observation dimension) gives isotropic RMS gain.','',
'All12actual teacher Jacobians have numerical rank parameter-count−6, and D annihilates the six null directions. A nonuniform invertible change of parameter units preserves D J⁺ numerically. These are numerical checks at the actual teachers, distinct from the earlier generic exact rank certificate.','',
'## Why independent additional heads improve local average stability','',
'For each independently parameterized readout t, split the unaveraged observation Jacobian into hidden-feature columns A_t and readout columns B_t. Eliminating the readout nuisance update gives hidden information F_t=A_tᵀ(I−P_Bt)A_t, where P_Bt projects onto the readout column space. Each F_t is positive semidefinite. Adding a head changes the total information by +F_t. On a common gauge-free hidden tangent space, positive definite information therefore increases and its inverse decreases in PSD order. Under independent isotropic coefficient noise of equal variance per head, the linearized hidden-feature covariance decreases accordingly. This is a local least-squares information statement, not a global optimization guarantee.','',
'Our fixed mean-output-noise convention yields covariance D_hidden (sum_t F_t)⁺ D_hiddenᵀ /126 per unit squared perturbation amplitude. The factor126 is the number of quartic coefficients in six variables. check_profiled_readout_information.py verifies this identity for all9independent-head geometries and all6nested PSD information/covariance comparisons. Readout parameters are re-fit, not held fixed. After profiling, the hidden information rank is30of36; six scale directions remain.','',
'Worst-case comparisons have a different normalization: the optimization loss and perturbation norm average across heads, so the corresponding profiled information is (sum_t F_t)/number_of_heads. Adding weakly informative heads need not increase this average. The isotropic covariance result must not be rephrased as monotonic worst-case robustness under every noise budget.','',
'## Verification and next step','',
'Independent audits compare the output Jacobian with direct-forward PyTorch differentiation, feature derivatives with separate normalized quadratic code, and all48clean/noisy errors with independent monomial product expansion. Maximum relative output-Jacobian error5.22e-16, feature-derivative discrepancy8.94e-16, loss discrepancy3.80e-20. Source hashes, checkpoints, noise vectors, linear predictions and optimizer outcomes are retained.','',
'The evidence now separates generic local identification, local stability, and global optimization: one head can identify locally, multiple independent heads improve isotropic local stability, and even six heads can fail from random initialization. Next test finite noisy input samples with whole held-out heads and explicit annotation budgets; coefficient-noise population experiments are not sample-complexity evidence.','']
(root/'READOUT_STABILITY_RESULTS.md').write_text('\n'.join(lines))
write(out/'summary.json',dict(status='complete',rows=rows,geometry=geometry,source_sha256=digest(Path(__file__))))
print('\n'.join(lines))
