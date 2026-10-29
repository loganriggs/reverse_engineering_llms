"""Audited block-curvature comparison; treat values below tolerance as ties."""
import json,statistics
from pathlib import Path
import numpy as np
from run_svhn_standard import digest,write
root=Path(__file__).parent;out=root/'results/bilinear_block_gauss_newton'
audit=json.loads((out/'audit.json').read_text());assert audit['status']=='passed' and len(audit['records'])==54
rows=[]
for a in audit['records']:
 p=out/(a['name']+'.json');assert digest(p)==a['result_sha256'] and digest(p.with_suffix('.npz'))==a['artifact_sha256']
 r=json.loads(p.read_text());rows.append(r)
 if r['method']=='full':
  old=root/'results/bilinear_gauss_newton'/f"{r['teacher']}_{r['seed']}_{r['student']}_0.001.npz"
  np.testing.assert_array_equal(np.load(old)['states'],np.load(p.with_suffix('.npz'))['states'])
lines=['# Alternating block Gauss–Newton on bilinear networks','','All 54 fits independently audited. Same 18 archived teacher/student/metric fixtures as full Gauss–Newton, damping .001, Armijo search, maximum 100 full-Jacobian evaluations, stop at population MSE <1e-12. Methods: full, forward bottom/top/readout, reverse readout/top/bottom. Each alternating iteration updates one block, not an entire sweep. Full Jacobians and Gram matrices are computed in every method; block methods solve smaller systems. This is an equal Jacobian-budget cap, not equal FLOPs, wall time, or specialized block implementation. No new hyperparameter search.','', '| Student | Method | Median final MSE (six teachers) | Below 1e-12 | Median Jacobian calls |','|---|---|---:|---:|---:|']
for student in ['tied_raw','tied_corrected','untied_raw']:
 for method in ['full','forward','reverse']:
  rs=[r for r in rows if r['student']==student and r['method']==method]
  lines.append(f"| {student} | {method} | {statistics.median(r['curves'][-1]['mse'] for r in rs):.6g} | {sum(r['curves'][-1]['mse']<1e-12 for r in rs)}/6 | {statistics.median(r['jacobian_evaluations'] for r in rs):g} |")
comparisons=[]
for method in ['forward','reverse']:
 wins=ties=losses=0
 for r in rows:
  if r['method']!=method:continue
  f=next(f for f in rows if f['method']=='full' and all(f[k]==r[k] for k in ['teacher','seed','student']))
  a,b=r['curves'][-1]['mse'],f['curves'][-1]['mse']
  tie=a<1e-12 and b<1e-12
  ties+=tie;wins+=not tie and a<b;losses+=not tie and a>=b
 comparisons.append(dict(method=method,wins=wins,ties=ties,losses=losses))
 lines.append(f'\n{method}: {wins} lower-error outcomes, {ties} both below stopping tolerance, {losses} higher-error outcomes versus full GN (18 paired fixtures).')
lines+=['','Full-method parameter trajectories reproduce all 18 archived .001-damping runs exactly. Every accepted state passes independent polynomial-coefficient Gaussian-moment loss replay; selected-block normal equations and Armijo descent pass; unselected parameters are unchanged. Initial/middle/final analytic Jacobians match autograd. All fits finite; no line-search failures.','',
'Interpretation: curvature-aware alternating updates can reach near-exact fits and improve some difficult tied fits, but do not show a universal advantage over full GN. Full GN solves all six untied-student cases; forward solves five, reverse six. Tiny differences below the stopping tolerance are not evidence of superiority. The timing totals are subsecond and single-run, so they are not reliable hardware performance benchmarks. These are function-fitting results, not circuit recovery.','',
'## Why alternating curvature steps can differ','',
'At a common state, let Q(delta)=rᵀJ delta + 1/2 deltaᵀ(JᵀJ+lambda M)delta, with lambda>0 and M positive definite. Full damped GN uniquely minimizes Q over all parameters. A block step minimizes the same Q over vectors supported on that block; consequently Q(full)<=Q(block)<=Q(0). Both nonzero steps are descent directions because g_Bᵀ delta_B=−delta_BᵀH_BB delta_B<0. This is a direct quadratic-minimization argument, not a nonlinear convergence theorem.','',
'After a block moves, the next block sees a different Jacobian and residual. Smaller restricted steps may remain accurate farther along a nonlinear trajectory, and line-search acceptance may differ. Thus local model dominance does not order final nonlinear errors. This identifies the relevant extension: measure linearization error and accepted step sizes before attributing block benefits to missing output directions.','',
'Decision: close this bounded block-order comparison. Next use a function-preserving gauge-balancing control to test conditioning directly, or multiple-readout supervision to address the separately demonstrated hidden-recovery failure. Do not expand optimizer sweeps merely to make every tied fit succeed.','']
(root/'BILINEAR_BLOCK_GAUSS_NEWTON_RESULTS.md').write_text('\n'.join(lines))
write(out/'summary.json',dict(status='complete',comparisons=comparisons,source_sha256=digest(Path(__file__))))
print('\n'.join(lines))
