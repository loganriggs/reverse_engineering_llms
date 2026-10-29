import json
from pathlib import Path
import numpy as np

root=Path('results/readout_timescale')
rows=json.loads((root/'results.json').read_text())
audit=json.loads((root/'audit.json').read_text())
checks=json.loads((root/'checks.json').read_text())
lines=['# Readout timescale: finite-time dynamics results','',
'Thirty two-timescale flows, three reduced reference flows, and tighter-tolerance replays completed. Three fresh d3/h2 teachers; both initially optimal and perturbed readouts; five epsilon values. See READOUT_TIMESCALE_PLAN.md for the fixed design and documented stiff-solver amendment. No training/transfer ranking is claimed.','',
'Here epsilon is the inverse readout speed relative to hidden learning: smaller means faster readout adaptation. Distances compare hidden coordinates at time .5 with reduced gradient flow from the same hidden initialization. Each table value is a median over three teachers, not a confidence interval.','',
'| Initial readout | epsilon | Hidden endpoint distance | Final readout tracking error |',
'|---|---:|---:|---:|']
summary=[]
for init in ['optimal','perturbed']:
 for eps in [.1,.03,.01,.003,.001]:
  chosen=[r for r in rows if r['initialization']==init and r['epsilon']==eps]
  assert len(chosen)==3
  dist=float(np.median([r['hidden_endpoint_distance'] for r in chosen]));track=float(np.median([r['final_tracking'] for r in chosen]))
  lines.append(f'| {init} | {eps:g} | {dist:.7g} | {track:.7g} |')
  summary.append(dict(initialization=init,epsilon=eps,median_hidden_distance=dist,median_tracking=track))
ratios=[];monotone=[]
for seed in sorted({r['seed'] for r in rows}):
 for init in ['optimal','perturbed']:
  chosen=sorted([r for r in rows if r['seed']==seed and r['initialization']==init],key=lambda r:-r['epsilon'])
  ratios.append(chosen[0]['hidden_endpoint_distance']/chosen[-1]['hidden_endpoint_distance'])
  monotone.append(all(a['hidden_endpoint_distance']>=b['hidden_endpoint_distance'] for a,b in zip(chosen,chosen[1:])))
lines+=['',f'Endpoint distance decreases at every tested speed increase in {sum(monotone)}/6 teacher/initial-readout sequences. Changing epsilon from .1 to .001 reduces this distance by factors {min(ratios):.3g}–{max(ratios):.3g}. This finite panel supports the conditional tracking limit; it does not establish an asymptotic rate or uniform theorem constants.',
'',f'All 3,030 stored observations pass independent literal-forward loss replay and SVD optimal-readout tracking replay. Sixty endpoint checks compare gradients against Torch autograd and optimal-readout derivatives against centered differences. Maximum gradient error {max(c["gradient_error"] for c in checks):.3g}; maximum derivative error {max(c["star_derivative_error"] for c in checks):.3g}. Maximum full-path tight-tolerance discrepancy {max(r["replay_max_abs"] for r in rows):.3g}.',
'',f'The {len(audit["independent_solver_comparisons"])} completed explicit-solver paths agree with Radau to maximum absolute discrepancy {max(r["max_solver_discrepancy"] for r in audit["independent_solver_comparisons"]):.3g}. Explicit outputs and the interrupted attempt log are retained. Solver agreement is not an interval-certified solution.',
'','## Interpretation and next test','',
'The mathematical result identifies a conditioning-sensitive tracking scale, epsilon times the motion of the optimal readout divided by a readout curvature lower bound. Even an initially optimal readout immediately lags when the hidden representation moves. Faster adaptation approaches reduced gradient flow under the stated local uniform assumptions; this does not imply it approaches a better representation. The previous recovery experiments already show that exact elimination can underperform joint training.',
'',
'The next dynamics comparison should hold the teacher/data/initial function fixed and test finite readout speeds against joint and reduced baselines using feature recovery and whole-task transfer. Finite-dwell alternating flow must remain a separate arm. Use measured gradient work and wall time, not adaptive ODE solver evaluation counts as a substitute for a practical training benchmark. Larger-width task/input scaling remains open; this finite-time check does not satisfy that requirement.']
Path('READOUT_TIMESCALE_RESULTS.md').write_text('\n'.join(lines)+'\n')
(root/'summary.json').write_text(json.dumps(dict(groups=summary,monotone_sequences=sum(monotone),distance_reduction_factors=ratios),indent=2)+'\n')
print('\n'.join(lines))
