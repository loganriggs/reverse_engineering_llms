"""Keep all optimization failures and distinguish supervision from identification."""
import json,statistics
from pathlib import Path
from run_svhn_standard import digest,write
root=Path(__file__).parent;out=root/'results/readout_recovery'
audit=json.loads((out/'audit.json').read_text());assert audit['status']=='passed' and len(audit['records'])==24
rows=[]
for a in audit['records']:
 p=out/(a['name']+'.json');assert digest(p)==a['result_sha256'] and digest(p.with_suffix('.npz'))==a['checkpoint_sha256'];rows.append(json.loads(p.read_text()))
lines=['# Readout information and component recovery: two-feature control','','24 exact-Gaussian-population fits: three independent teachers, two initializations, four supervision arms. Four input dimensions, two bilinear hidden features, linear readout over their three unordered products. This top parameterization is different from the previous factorized C/D/w architecture. All fits use the same SciPy trust-region least-squares settings, analytic coefficient Jacobian, and maximum200 function evaluations. No restarts.','',
'One/two/three arms expose nested independent scalar heads. The intervention arm exposes the first head plus outputs after ablating hidden feature0 or1; the student uses the corresponding labeled masks with the same underlying readout. These are privileged teacher interventions with known node correspondence, not interventions inferred from passive data. Targets are individually Gaussian-RMS normalized, and the objective averages across supervised outputs.','',
'Four further readouts are entirely withheld from representation training. After freezing the learned hidden product features, fit those heads using32 or128 labels, and also compute the exact population-projection floor. No held-out labels choose representation, stopping, hyperparameters or restarts. This noiseless low-dimensional transfer is deliberately simple.','',
'| Supervision | Parameters | Outputs | Function MSE<1e-12 | Both component cosines>1−1e-6 | Max final source MSE |','|---|---:|---:|---:|---:|---:|']
for arm in ['one','two','three','interventions']:
 rs=[r for r in rows if r['arm']==arm]
 lines.append(f"| {arm} | {rs[0]['parameters']} | {rs[0]['supervised_outputs']} | {sum(r['mse']<1e-12 for r in rs)}/6 | {sum(min(r['component_cosines'])>1-1e-6 for r in rs)}/6 | {max(r['mse'] for r in rs):.6g} |")
lines+=['','These are six fits but only three independent teacher draws. Parameter counts and supervision differ, so this is an information intervention, not a parameter-matched architecture comparison. Each near-exact fit also recovers both components and transfers nearly exactly. Failed fits remain visible below rather than being hidden by median transfer error.','',
'| Teacher | Initialization | Arm | Source MSE | Minimum matched component cosine | Mean held-out projection floor | Mean held-out MSE,128 labels |','|---|---|---|---:|---:|---:|---:|']
for r in sorted(rows,key=lambda r:r['name']):
 if r['mse']<1e-12:continue
 lines.append(f"| {r['teacher_seed']} | {r['initial_seed']} | {r['arm']} | {r['mse']:.6g} | {min(r['component_cosines']):.6g} | {statistics.mean(r['transfer_mse']['population']):.6g} | {statistics.mean(r['transfer_mse']['128']):.6g} |")
lines+=['','## What changed our interpretation','',
'A single output recovers components in4/6fits, three outputs and interventions in5/6each, while two outputs succeed in3/6. These small counts are not evidence of a monotone advantage from extra supervision. The new [two-feature scalar-identification argument](TWO_FEATURE_SCALAR_IDENTIFICATION.md) explains why scalar recovery is possible: a generic binary-quadratic readout factors into two rank-four quadratics, identifying the feature pencil; only the original feature lines in that pencil are bilinear. All three teachers satisfy the independence and nondegeneracy assumptions.','',
'This control therefore does not test removal of a generic scalar-output identification ambiguity. Its remaining failures are approximate/local optimization outcomes, not demonstrated non-identifiability. Extra outputs can change the optimization landscape, but larger residuals are not an information-theoretic floor. The symbolic proof has no approximate stability guarantee.','',
'Verification: independently generated monomial products reproduce all fitted coefficient maps; initial/final analytic Jacobians agree with PyTorch differentiation of direct forward evaluation on31fresh inputs. Gaussian coefficient moments replay losses; separate81-node quadrature replays hidden-subspace and component comparisons. All held-out readouts satisfy normal equations, and exact Gaussian transfer errors replay. Source hashes and all24checkpoints are recorded. SciPy termination status0 (budget exhausted) is retained even when function error is tiny; optimizer termination flags do not define recovery.','',
'Next: extend to three hidden features in six inputs, comparing1/3/6 independent readouts and explicitly labeled ablations. The existing full-product-span identifiability theorem applies to six spanning readouts, while the two-feature scalar factorization argument no longer automatically applies. Preserve exact population objectives and whole held-out heads so optimization, information, and transfer remain separately measurable.','']
(root/'READOUT_RECOVERY_RESULTS.md').write_text('\n'.join(lines))
write(out/'summary.json',dict(status='complete',rows=rows,source_sha256=digest(Path(__file__))))
print('\n'.join(lines))
