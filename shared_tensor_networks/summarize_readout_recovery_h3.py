"""Keep all optimization failures and distinguish supervision from identification."""
import json,statistics
from pathlib import Path
from run_svhn_standard import digest,write
root=Path(__file__).parent;out=root/'results/readout_recovery_h3'
audit=json.loads((out/'audit.json').read_text());assert audit['status']=='passed' and len(audit['records'])==24
rows=[]
for a in audit['records']:
 p=out/(a['name']+'.json');assert digest(p)==a['result_sha256'] and digest(p.with_suffix('.npz'))==a['checkpoint_sha256'];rows.append(json.loads(p.read_text()))
lines=['# Three-feature readout recovery','','24 exact-population fits: three fresh teachers218001–3 and two initializations219001–2, three bilinear hidden features in six inputs. One/three/six nested independent output heads, or one head plus three individually labeled hidden-node ablations. SciPyTRF with analytic coefficient Jacobian, maximum200function evaluations, fixed1e-12tolerances, no restarts. The six-head teachers span all six hidden products and have six independent input factors; the auditor verifies these theorem assumptions.','',
'Each target is Gaussian-RMS normalized and loss averages over supervised outputs. Four other heads remain excluded from representation training; afterward fit their readouts on frozen hidden products using32/128labels, plus an exact population projection floor. Intervention masks have known node correspondence and are privileged supervision. Top readouts are linear over hidden products, unlike the earlier factorized C/D/w architecture.','',
'| Supervision | Parameters | Outputs | Function MSE<1e-12 | All component cosines>1−1e-6 | Max final source MSE |','|---|---:|---:|---:|---:|---:|']
for arm in ['one','three','six','interventions']:
 rs=[r for r in rows if r['arm']==arm]
 lines.append(f"| {arm} | {rs[0]['parameters']} | {rs[0]['supervised_outputs']} | {sum(r['mse']<1e-12 for r in rs)}/6 | {sum(min(r['component_cosines'])>1-1e-6 for r in rs)}/6 | {max(r['mse'] for r in rs):.6g} |")
lines+=['','These are six fits per arm but only three independent teachers. More supervision changes parameter count, residual dimension and optimizer work; this is not an equal-compute or parameter-matched comparison. Every near-exact fit recovers all three components and transfers nearly exactly. The remaining outcomes are retained below.','',
'| Teacher | Initialization | Arm | Source MSE | Minimum matched component cosine | Mean held-out projection floor | Mean held-out MSE,128 labels |','|---|---|---|---:|---:|---:|---:|']
for r in sorted(rows,key=lambda r:r['name']):
 if r['mse']<1e-12:continue
 lines.append(f"| {r['teacher_seed']} | {r['initial_seed']} | {r['arm']} | {r['mse']:.6g} | {min(r['component_cosines']):.6g} | {statistics.mean(r['transfer_mse']['population']):.6g} | {statistics.mean(r['transfer_mse']['128']):.6g} |")
lines+=['','## Interpretation','',
'Three heads and labeled ablations recover6/6, versus3/6for one head and5/6for six heads. This supports a useful supervision effect on these tasks but not a monotone benefit of more heads or a broad statistical claim. A six-head fit fails despite the exact-fit identifiability theorem: information sufficiency does not guarantee optimizer success.','',
'The [local-rank certificate](READOUT_LOCAL_IDENTIFICATION.md) adds a crucial qualification: even one scalar head is generically locally identifiable modulo six scaling freedoms in this d6/h3family. Thus these scalar failures cannot establish a universal continuous identification ambiguity. The certificate does not rule out distant discrete alternatives or certify the precise random teachers.','',
'Independent audits reproduce teacher/learned coefficient maps, initial/final Jacobians using direct-forward PyTorch differentiation, all population losses, component matching with separate729-node quadrature, and held-out regression normal equations/errors. Maximum relative Jacobian discrepancy8.58e-16; loss discrepancy2.78e-16. All24checkpoints, failed/budget-limited outcomes, seeds, source hashes and optimizer diagnostics are retained.','',
'Next: distinguish stability from mere exact identification. Evaluate coefficient-Jacobian spectra after removing scale directions, and test small controlled label perturbations while holding the teacher and optimization basin fixed. This can test whether additional heads constrain weak directions more strongly, without mistaking convergence failures for non-identifiability. Further random restarts to rescue a favored arm are not warranted.','']
(root/'READOUT_RECOVERY_H3_RESULTS.md').write_text('\n'.join(lines))
write(out/'summary.json',dict(status='complete',rows=rows,source_sha256=digest(Path(__file__))))
print('\n'.join(lines))
