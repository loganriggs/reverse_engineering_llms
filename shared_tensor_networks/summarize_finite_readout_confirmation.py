"""Prespecified teacher-level confirmation; average all optimizer seeds, no selection."""
import json,statistics
from pathlib import Path
from run_svhn_standard import digest,write
root=Path(__file__).parent;out=root/'results/finite_readout_confirmation';audit=json.loads((out/'audit.json').read_text());assert audit['status']=='passed' and len(audit['records'])==120
rows=[]
for a in audit['records']:
 p=out/(a['name']+'.json');assert digest(p)==a['result_sha256'] and digest(p.with_suffix('.npz'))==a['checkpoint_sha256'];rows.append(json.loads(p.read_text()))
teachers=[222001,222002,222003,222004,222005];aggregates=[]
for budget in [384,1536]:
 for initialization in ['random','teacher']:
  for seed in teachers:
   for heads in [1,3,6]:
    selected=[r for r in rows if r['teacher_seed']==seed and r['budget']==budget and r['heads']==heads and r['initialization'].startswith(initialization)]
    assert len(selected)==(3 if initialization=='random' else 1)
    aggregates.append(dict(budget=budget,initialization=initialization,teacher_seed=seed,heads=heads,
      transfer=statistics.mean(statistics.mean(r['transfer_mse']['learned_128']) for r in selected),oracle=statistics.mean(statistics.mean(r['transfer_mse']['oracle_128']) for r in selected),source=statistics.mean(statistics.mean(r['population_mse']) for r in selected),
      recovered_fraction=statistics.mean(float(min(r['component_cosines'])>.99) for r in selected),budget_stops=sum(r['status']==0 for r in selected)))
lines=['# Fresh-teacher finite-label confirmation','','All120fits and independent audits complete. Five fresh teachers222001–222005, three random initializations223001–223003 per condition, and a separate teacher-start diagnostic. Prespecified recipe and endpoint are in FINITE_READOUT_CONFIRMATION_PLAN.md. Source budgets384/1536, heads1/3/6, label noise.05, fixed200evaluation cap. No restarts, best-seed selection, added teachers, or development-teacher pooling.','',
'Primary endpoint: average Gaussian transfer MSE over four whole held-out heads after128noisy labels/head, then average all three random initializations within each teacher. Five teacher aggregates are the independent units; individual task/initialization records are not extra teacher replications. Teacher starts are privileged diagnostics, not a practical method.','',
'| Source labels | Initialization | Heads | Median teacher transfer MSE (range) | Mean fraction with all component cosines>.99 | Budget stops |','|---|---|---:|---:|---:|---:|']
for budget in [384,1536]:
 for init in ['random','teacher']:
  for heads in [1,3,6]:
   a=[r for r in aggregates if r['budget']==budget and r['initialization']==init and r['heads']==heads];values=[r['transfer'] for r in a]
   lines.append(f"| {budget} | {init} | {heads} | {statistics.median(values):.6g} ({min(values):.6g}–{max(values):.6g}) | {statistics.mean(r['recovered_fraction'] for r in a):.3f} | {sum(r['budget_stops'] for r in a)} |")
lines+=['','## Every teacher comparison','','| Source labels | Initialization | Teacher | One head | Three heads | Six heads | Oracle |','|---|---|---|---:|---:|---:|---:|']
comparisons=[]
for budget in [384,1536]:
 for init in ['random','teacher']:
  wins={3:0,6:0}
  for seed in teachers:
   a={r['heads']:r for r in aggregates if r['budget']==budget and r['initialization']==init and r['teacher_seed']==seed}
   for h in [3,6]:wins[h]+=a[h]['transfer']<a[1]['transfer']
   lines.append(f'| {budget} | {init} | {seed} | '+' | '.join(f"{a[h]['transfer']:.6g}" for h in [1,3,6])+f" | {a[1]['oracle']:.6g} |")
  comparisons.append(dict(budget=budget,initialization=init,three_beats_one=wins[3],six_beats_one=wins[6]))
lines+=['','## Prespecified paired outcomes','']
for c in comparisons:lines.append(f"- {c['budget']}labels, {c['initialization']} initialization: three beats one in{c['three_beats_one']}/5teachers; six beats one in{c['six_beats_one']}/5.")
lines+=['',
'At1536source labels, six heads beat one for all five fresh teacher aggregates, with median held-out MSE .000413 versus .123. Three heads win4/5. At384labels, six heads win only2/5and have worse median transfer than one. The confirmation therefore supports a budget-dependent task-diversity benefit, not the development impression that three heads should be the preferred allocation. It does not show monotonicity: six versus three is mixed, and one six-head teacher aggregate remains .073 at the larger budget.','',
'Component recovery remains imperfect: at1536labels, all three component cosines exceed.99in11/15random six-head fits, versus8/15three-head and2/15one-head fits. The exact-teacher six-head diagnostic meets that threshold for5/5teachers. This separates practical random-initialization robustness from local statistical performance.','',
'![Fresh teacher transfer at fixed source-label budgets](figures/finite_readout_confirmation.png)','',
'These are transfer outcomes at equal source scalar-label budgets, not equal input counts, parameter counts, or computational work. Four held-out128-label tasks add512labels equally to every allocation. TeacherRMSnormalization is a privileged simulation convention. Learning from three output heads need not be uniformly best, and the comparison cannot establish a universally optimal label allocation.','',
'Independent audits replay seeded teacher/input/noise generation, nested budgets, all initializations, direct-forward Jacobians, polynomial population losses,729-node component matching, polynomial baseline and adaptation normal equations. All unsuccessful and budget-limited fits are retained;32-label adaptation and all source-fit baselines remain in JSON.','',
'Decision: close this fixed confirmation at120fits. Interpret mixed/negative outcomes without altering the recipe. Separate transfer improvement, component recovery and random-initialization robustness before planning any larger-network extension.','']
(root/'FINITE_READOUT_CONFIRMATION_RESULTS.md').write_text('\n'.join(lines))
write(out/'summary.json',dict(status='complete',aggregates=aggregates,comparisons=comparisons,source_sha256=digest(Path(__file__))))
print('\n'.join(lines))
