"""All teacher blocks, fixed ten-dimensional output-span adaptation."""
import json,statistics
from pathlib import Path
from run_svhn_standard import digest,write
root=Path(__file__).parent;out=root/'results/shared_separate_readouts';audit=json.loads((out/'audit.json').read_text());assert audit['status']=='passed' and len(audit['records'])==45
rows=[]
for a in audit['records']:
 p=out/(a['name']+'.json');assert digest(p)==a['result_sha256'] and digest(p.with_suffix('.npz'))==a['checkpoint_sha256'];rows.append(json.loads(p.read_text()))
aggs=[]
for seed in range(225001,225006):
 for method in ['shared','separate_width','separate_budget']:
  a=[r for r in rows if r['teacher_seed']==seed and r['method']==method];assert len(a)==3
  counts=[]
  for r in a:
   if method=='shared':counts.append(json.loads((root/r['input_file']).with_suffix('.json').read_text())['nfev'])
   else:counts.append(sum(h['nfev'] for h in r['head_records']))
  aggs.append(dict(seed=seed,method=method,parameters=a[0]['parameters'],source=statistics.mean(statistics.mean(r['source_mse']) for r in a),transfer=statistics.mean(statistics.mean(r['transfer_mse']['learned_128']) for r in a),floor=statistics.mean(statistics.mean(r['transfer_mse']['learned_population']) for r in a),nfev_mean=statistics.mean(counts)))
lines=['# Shared versus independent task dictionaries','','300new independent-head fits, grouped into30models, compared with15archived shared models. Five d8/h4teachers, three paired initializations each, ten supervised tasks,512common input examples and5120source scalar labels, noise.05. Protocol fixed in SHARED_SEPARATE_READOUT_PLAN.md. These teachers were already used in scaling; this is a mechanistic control, not a new independent teacher confirmation.','',
'Sharedh4has164parameters. Ten independenth4heads have740parameters and start with exactly the same source predictions as the shared model. Ten independenth1heads have170parameters, the nearest integer uniform-width budget; their initial function and capacity differ. Each independent head has its own200evaluationTRFcap. Independent optimization changes the trust-region schedule and total computation even though the objective is separable. This does not isolate tying alone or provide a compute-matched optimizer comparison.','',
'All methods adapt through exactly ten predicted source outputs, normalized to unit GaussianRMS. Independent models do not receive100hidden-product transfer features. Four whole held-out heads receive the same32/128noisy labels; population projection floors and oracle source-output features are retained. Unit population normalization is privileged synthetic information but uses no held-out labels.','',
'| Method | Parameters | Median teacher source MSE | Median teacher transfer MSE,128labels (range) | Median population transfer floor |','|---|---:|---:|---:|---:|']
for method in ['shared','separate_width','separate_budget']:
 a=[r for r in aggs if r['method']==method];v=[r['transfer'] for r in a]
 lines.append(f"| {method} | {a[0]['parameters']} | {statistics.median(r['source'] for r in a):.6g} | {statistics.median(v):.6g} ({min(v):.6g}–{max(v):.6g}) | {statistics.median(r['floor'] for r in a):.6g} |")
lines+=['','Entries first average four adaptation tasks and three initialization outcomes within each teacher. No best-head or best-seed selection.','',
'| Teacher | Shared transfer | Independenth4transfer | Independenth1transfer |','|---|---:|---:|---:|']
wins={m:0 for m in ['separate_width','separate_budget']}
for seed in range(225001,225006):
 a={r['method']:r for r in aggs if r['seed']==seed}
 for m in wins:wins[m]+=a['shared']['transfer']<a[m]['transfer']
 lines.append(f'| {seed} | '+' | '.join(f"{a[m]['transfer']:.6g}" for m in ['shared','separate_width','separate_budget'])+' |')
lines+=['',f"Shared beats independenth4in{wins['separate_width']}/5teacher blocks and independenth1in{wins['separate_budget']}/5. These counts apply to the prescribed output-span transfer endpoint, not every possible use of the learned representations.",'',
'## What this control can and cannot establish','',
'There is direct evidence of optimization failure in13/15independent equal-width fits: copying the corresponding learned shared dictionary into each independent head gives a strictly lower empirical loss. check_shared_solution_embedding.py verifies all copied predictions and loss values. With identical data, normalization and no added parameter penalty, the independent hypothesis class contains the shared class, so its global minimum training loss cannot be higher. These fitted independent solutions are therefore demonstrably not global empirical minima. This does not prove all of the transfer gap is optimization, nor establish the shared solutions as global minima.','',
'The comparison measures the benefit of fitting a common dictionary jointly versus fitting each task independently, under the implemented optimization protocols. The wider independent model has more parameters and may overfit or optimize differently. The near-budget model is much more restricted per task. Shared-versus-separate differences cannot be assigned uniquely to parameter count, regularization, basin geometry or reuse without further controls.','',
'Population projection floors distinguish an inadequate span of learned source functions from finite adaptation-label noise. Low training-task error or a strong output span is not itself evidence that hidden components match the teacher. The shared models already have separate component-recovery measurements; this experiment does not identify a physical execution DAG.','',
'Independent audits verify provenance against the prior scaling audit, every initial/final independent-head derivative against direct autograd, coefficient predictions on400fresh inputs, all source losses, normalization, and transfer least-squares normal equations and exact Gaussian errors. All45groups and300individual head fits are retained; maximum relative Jacobian discrepancy1.03e-15. Source hashes and every head status/evaluation count are saved.','',
'Decision: close this fixed comparison. Use the hypothesis ledger to separate statistical sharing effects from optimization dynamics; do not rescue independent or shared fits with selected retries.','']
(root/'SHARED_SEPARATE_READOUT_RESULTS.md').write_text('\n'.join(lines));write(out/'summary.json',dict(status='complete',aggregates=aggs,wins=wins,source_sha256=digest(Path(__file__))))
print('\n'.join(lines))
