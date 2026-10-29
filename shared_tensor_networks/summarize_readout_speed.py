import json
from pathlib import Path
import numpy as np

ROOT=Path('results/readout_speed');rows=json.loads((ROOT/'results.json').read_text());audit=json.loads((ROOT/'audit.json').read_text())
seeds=sorted({r['seed'] for r in rows});groups=list(dict.fromkeys((r['method'],r['speed']) for r in rows));stats=[]
def teacher_means(method,speed,key):
 return np.array([np.mean([r[key] for r in rows if r['seed']==seed and (r['method'],r['speed'])==(method,speed)]) for seed in seeds])
lines=['# Finite readout speed: feature recovery and transfer','',
'All70fits completed their2000sweep budget; none failed its line search. These are finite-budget outcomes, not convergence claims. Five archived teachers, two starts each, seven predeclared methods; shared initial factors and optimal readout. This is a paired follow-up on earlier tasks, not a fresh-teacher replication. See READOUT_SPEED_RECOVERY_PLAN.md.','',
'Each error is the median across five teacher means, with two starts averaged within teacher; transfer also averages four unseen source-task identities, each adapted using128labels. Recovery means every matched teacher/student quadratic feature has absolute Gaussian cosine>.99.','',
'| Method | Readout speed | Source population MSE | Transfer MSE | All4features | Accepted hidden clock range |',
'|---|---:|---:|---:|---:|---:|']
for method,speed in groups:
 a=[r for r in rows if (r['method'],r['speed'])==(method,speed)]
 s=dict(method=method,speed=speed,source=float(np.median(teacher_means(method,speed,'source_mse'))),transfer=float(np.median(teacher_means(method,speed,'transfer_mse'))),recovery=sum(min(r['cosines'])>.99 for r in a),clock_min=min(r['hidden_clock'] for r in a),clock_max=max(r['hidden_clock'] for r in a),median_seconds=float(np.median([r['seconds'] for r in a])),gradient_calls=sum(r['gradient_calls'] for r in a),forward_calls=sum(r['forward_calls'] for r in a),readout_solves=sum(r['readout_solves'] for r in a),flow_calls=sum(r['flow_calls'] for r in a))
 stats.append(s);lines.append(f'| {method} | {speed if speed else "exact optimum"} | {s["source"]:.6g} | {s["transfer"]:.6g} | {s["recovery"]}/10 | {s["clock_min"]:.4g}–{s["clock_max"]:.4g} |')
wins={}
for comparator in [('joint',1),('reduced',0),('split',100)]:
 wins[str(comparator)]=int(np.sum(teacher_means('split',10,'transfer_mse')<teacher_means(*comparator,'transfer_mse')))
lines+=['','The split method takes a retracted hidden gradient step, then integrates the frozen-hidden readout flow exactly. It is not exact integration of both alternating blocks. Reduced steps solve the readout optimally at each candidate hidden state. Joint methods use simultaneous explicit gradient steps. All three differ from the earlier TRF/Gauss–Newton protocols.',
'',f'Split speed10 has the lowest aggregate transfer median here; it beats joint speed1 on {wins[str(("joint",1))]}/5 teacher means, reduced on {wins[str(("reduced",0))]}/5, and split speed100 on {wins[str(("split",100))]}/5. The best median is descriptive across a predeclared panel, not a validated optimizer-selection rule. Failures and both starts remain included.',
'','## Stability and the timescale confound','',
'For fixed hidden features, write H=(Phi^T Phi+lambda I)/N, U*=H^{-1}Phi^T Y/N, and e=U-U*. An explicit readout update gives e_next=(I-tau H)e. In an eigenvector with eigenvalue h, its readout objective contribution is multiplied by (1-tau h)^2; nonincrease for every e requires0<=tau<=2/lambda_max(H). Exact readout flow gives e_next=exp(-tau H)e and multiplier exp(-2tau h), stable for every tau>=0. This is a fixed-hidden quadratic statement, not global hidden-network convergence.',
'',
'In the joint speed100 arm, backtracking reduces accumulated hidden time to roughly.59–1.55 versus20 for joint speed1, all split arms and reduced. Hence equal sweep counts do not compare equal hidden optimization time. The bad fast-joint outcomes cannot be interpreted simply as evidence against fast readout adaptation. Exact readout flow removes this particular explicit-block stability restriction, but finite splitting and different trajectories remain.',
'','## Verification and resources','',
f'Independent audit replays all{audit["states"]}stored states via literal forward objectives; final source and transfer use Wick moments independently of producer polynomial coefficients. Quadratic coefficient matching, SVD adaptation, equal initial states, scalar loss monotonicity and accepted clocks pass. All70final gradients independently agree with Torch autograd (max absolute error {max(c["gradient_max_error"] for c in audit["checks"]):.3g}); nine separate fixtures check exact readout flow against an augmented matrix exponential. Internal line-search decisions are not all independently replayed.',
'',
'Each fit uses2000gradient evaluations. Reduced methods additionally solve readouts for candidate states, split methods use2000eigendecompositions/exact block flows, and joint methods pay extra forward evaluations for backtracking. Single-run CPU times and all call counts are saved in summary.json; subsecond timings are descriptive and unsuitable for a hardware speed ranking. No GPU was used.',
'','## Decision','',
'This supports testing moderate finite readout adaptation, but does not establish its superiority. Next use fresh teachers and match accepted hidden-time budgets for joint arms, with measured compute reported separately; include whole-task transfer and feature matching. Do not tune these same starts until every method recovers. Larger d64/h32 scaling remains open.']
Path('READOUT_SPEED_RECOVERY_RESULTS.md').write_text('\n'.join(lines)+'\n');(ROOT/'summary.json').write_text(json.dumps(dict(groups=stats,split10_teacher_wins=wins),indent=2)+'\n');print('\n'.join(lines))
