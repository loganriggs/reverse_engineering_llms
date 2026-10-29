import json
from pathlib import Path
import numpy as np

ROOT=Path('results/readout_speed_confirmation');rows=json.loads((ROOT/'results.json').read_text());audit=json.loads((ROOT/'audit.json').read_text())
assert len(rows)==70
seeds=sorted({r['seed'] for r in rows});groups=list(dict.fromkeys((r['method'],r['speed']) for r in rows))
def means(method,speed,key):
 return np.array([np.mean([r[key] for r in rows if r['seed']==seed and (r['method'],r['speed'])==(method,speed)]) for seed in seeds])
stats=[];matched=sum(r['status']=='clock_reached' for r in rows)
lines=['# Fresh-teacher readout-speed confirmation','',
f'All 70 predeclared fits are terminal; {matched}/70 reached the target accumulated hidden-step time of 20. Five fresh teachers and two fresh initializations per teacher. This panel is separate from the previous archived-teacher results. Same seven methods, fixed rates and line-search settings; no rescue or tuning. See READOUT_SPEED_CONFIRMATION_PLAN.md.',
'','Each error is a median of five teacher means, averaging both starts before taking the median. Transfer also averages four whole-task holdouts, each adapted with 128 noisy labels. Recovery requires all four matched quadratic features to exceed absolute Gaussian cosine .99. Failed or capped cases are retained in all aggregates.','',
'| Method | Readout speed | Source MSE | Transfer MSE | All features recovered | Clock reached | Median gradient calls |',
'|---|---:|---:|---:|---:|---:|---:|']
for method,speed in groups:
 a=[r for r in rows if (r['method'],r['speed'])==(method,speed)]
 s=dict(method=method,speed=speed,source=float(np.median(means(method,speed,'source_mse'))),transfer=float(np.median(means(method,speed,'transfer_mse'))),recovery=sum(min(r['cosines'])>.99 for r in a),matched=sum(r['status']=='clock_reached' for r in a),min_clock=min(r['hidden_clock'] for r in a),max_clock=max(r['hidden_clock'] for r in a),median_gradient_calls=float(np.median([r['gradient_calls'] for r in a])),median_seconds=float(np.median([r['seconds'] for r in a])),total_forward_calls=sum(r['forward_calls'] for r in a),total_readout_solves=sum(r['readout_solves'] for r in a),total_flow_calls=sum(r['flow_calls'] for r in a))
 stats.append(s);lines.append(f'| {method} | {speed if speed else "exact optimum"} | {s["source"]:.6g} | {s["transfer"]:.6g} | {s["recovery"]}/10 | {s["matched"]}/10 | {s["median_gradient_calls"]:g} |')
wins={}
for other in [('joint',1),('joint',10),('joint',100),('reduced',0),('split',100)]:
 delta=means('split',10,'transfer_mse')-means(*other,'transfer_mse')
 wins[str(other)]=dict(wins=int(np.sum(delta<0)),teacher_differences=delta.tolist())
lines+=['','## Paired comparisons','']
for other,item in wins.items():lines.append(f'- Split speed 10 has lower teacher-mean transfer than {other} on {item["wins"]}/5 fresh teachers.')
lines+=['','The earlier moderate-speed advantage did not replicate: split10 beats joint1 on only 2/5 fresh teachers and reduced on only 1/5, with worse median transfer than both. Joint1, split10 and reduced all had hidden clock20 in BOTH panels, so this primary reversal is not explained by the new clock matching. Teachers and initializations both changed; this experiment does not isolate which caused the reversal.',
'','At a common speed, the aggregate joint and split errors are similar here, while split updates use fewer gradients at high speed. This suggests a numerical integration benefit for the stiff readout block, not a demonstrated benefit for recovering reusable computation. Single-run timings and different matrix-solve costs prevent a general compute-efficiency claim.',
'','These counts and the aggregate medians do not constitute a universal method ordering. The five teachers are the independent experimental blocks; the two initializations and four holdouts are not additional independent teacher replications. The predeclared primary comparisons are split10 versus joint1 and reduced.','',
'## What the clock control does and does not remove','',
'Matching the sum of accepted hidden step sizes removes the earlier gross reduction in hidden-learning time when fast explicit updates backtracked. It does not equalize compute, wall time, discretization error or physical time in a switched ODE. Joint methods update both blocks explicitly; split methods retract hidden factors then integrate only the frozen-hidden readout exactly. Reduced methods solve the readout at each candidate state. These are finite-step algorithms, not error-controlled full ODE integrations.',
'',
'Per-fit elapsed CPU time, gradient and forward evaluations, readout solves, exact block-flow calls, attained clocks and all statuses are retained. Timing is one run on one pinned CPU core, not a robust hardware benchmark. The target clock is a finite budget, not a convergence certificate.',
'','## Verification','',
f'Independent audit checks {audit["fits"]} checkpoints and {audit["states"]} stored states, fresh source/holdout target and noise generation, teacher Gaussian RMS, source-data hashes, common initial states, gradient/resource counters and the last shortened step. Literal-forward objective replay, SVD adaptation, Wick source/transfer moments, and quadratic coefficient matching pass. All final gradients agree with independent Torch autograd; maximum absolute error {max(c["gradient_max_error"] for c in audit["checks"]):.3g}. All internal line-search decisions are not independently replayed.',
'','## Decision','',
'Close this bounded fresh-teacher/clock control; do not expand a rate search on these teachers. Preserve the full method table rather than promoting the best aggregate as a universal optimizer. The next broad-goal experiment is the outstanding d64/h32 task/input scaling grid, retaining both data coverage and optimization diagnostics. Native MLP research remains paused; the overall toy-research goal remains active.']
Path('READOUT_SPEED_CONFIRMATION_RESULTS.md').write_text('\n'.join(lines)+'\n');(ROOT/'summary.json').write_text(json.dumps(dict(groups=stats,paired=wins,matched=matched),indent=2)+'\n');print('\n'.join(lines))
