"""Prespecified teacher-level confirmation; average all optimizer seeds, no selection."""
import json,statistics
from pathlib import Path
from run_svhn_standard import digest,write
root=Path(__file__).parent;out=root/'results/finite_readout_h4';audit=json.loads((out/'audit.json').read_text());assert audit['status']=='passed' and len(audit['records'])==120
rows=[]
for a in audit['records']:
 p=out/(a['name']+'.json');assert digest(p)==a['result_sha256'] and digest(p.with_suffix('.npz'))==a['checkpoint_sha256'];rows.append(json.loads(p.read_text()))
teachers=[225001,225002,225003,225004,225005];aggregates=[]
for budget in [1280,5120]:
 for initialization in ['random','teacher']:
  for seed in teachers:
   for heads in [1,4,10]:
    selected=[r for r in rows if r['teacher_seed']==seed and r['budget']==budget and r['heads']==heads and r['initialization'].startswith(initialization)]
    assert len(selected)==(3 if initialization=='random' else 1)
    aggregates.append(dict(budget=budget,initialization=initialization,teacher_seed=seed,heads=heads,
      transfer=statistics.mean(statistics.mean(r['transfer_mse']['learned_128']) for r in selected),oracle=statistics.mean(statistics.mean(r['transfer_mse']['oracle_128']) for r in selected),source=statistics.mean(statistics.mean(r['population_mse']) for r in selected),
      recovered_fraction=statistics.mean(float(min(r['component_cosines'])>.99) for r in selected),budget_stops=sum(r['status']==0 for r in selected)))
lines=['# Four-feature fixed-budget scaling','','All120fits and independent audits complete. Five fresh teachers225001–225005, three random initializations226001–226003 per condition, and a separate teacher-start diagnostic. Prespecified recipe and endpoint are in FINITE_READOUT_H4_PLAN.md. Source budgets1280/5120, heads1/4/10, label noise.05, fixed200evaluation cap. No restarts, best-seed selection, added teachers, or development-teacher pooling.','',
'Primary endpoint: average Gaussian transfer MSE over four whole held-out heads after128noisy labels/head, then average all three random initializations within each teacher. Five teacher aggregates are the independent units; individual task/initialization records are not extra teacher replications. Teacher starts are privileged diagnostics, not a practical method.','',
'| Source labels | Initialization | Heads | Median teacher transfer MSE (range) | Mean fraction with all component cosines>.99 | Budget stops |','|---|---|---:|---:|---:|---:|']
for budget in [1280,5120]:
 for init in ['random','teacher']:
  for heads in [1,4,10]:
   a=[r for r in aggregates if r['budget']==budget and r['initialization']==init and r['heads']==heads];values=[r['transfer'] for r in a]
   lines.append(f"| {budget} | {init} | {heads} | {statistics.median(values):.6g} ({min(values):.6g}–{max(values):.6g}) | {statistics.mean(r['recovered_fraction'] for r in a):.3f} | {sum(r['budget_stops'] for r in a)} |")
lines+=['','## Every teacher comparison','','| Source labels | Initialization | Teacher | One head | Four heads | Ten heads | Oracle |','|---|---|---|---:|---:|---:|---:|']
comparisons=[]
for budget in [1280,5120]:
 for init in ['random','teacher']:
  wins={4:0,10:0}
  for seed in teachers:
   a={r['heads']:r for r in aggregates if r['budget']==budget and r['initialization']==init and r['teacher_seed']==seed}
   for h in [4,10]:wins[h]+=a[h]['transfer']<a[1]['transfer']
   lines.append(f'| {budget} | {init} | {seed} | '+' | '.join(f"{a[h]['transfer']:.6g}" for h in [1,4,10])+f" | {a[1]['oracle']:.6g} |")
  comparisons.append(dict(budget=budget,initialization=init,four_beats_one=wins[4],ten_beats_one=wins[10]))
lines+=['','## Prespecified paired outcomes','']
for c in comparisons:lines.append(f"- {c['budget']}labels, {c['initialization']} initialization: four beats one in{c['four_beats_one']}/5teachers; ten beats one in{c['ten_beats_one']}/5.")
lines+=['',
'The budget-dependent pattern persists at the larger size. At5120labels, ten heads beat one for5/5teachers and achieve median transferMSE.000852versus.310; all four component cosines exceed.99in13/15random ten-head fits versus5/15one-head and7/15four-head fits. At1280labels, ten heads win only2/5, whereas four heads win5/5. The preferred allocation therefore depends on budget and endpoint.','',
'Teacher-initialized controls meet the component threshold in every condition, with much smaller transfer errors than many random starts. This demonstrates accessible good solutions and leaves basin-finding as an important practical limitation; it does not certify global optima or prove every failure has the same cause.','',
'Scaling context: four hidden bilinear features in eight input dimensions, versus three in six previously. Ten independent heads can span all ten unordered hidden products. Both model size and label budgets increase; this does not isolate the effect of dimension alone. The direct sample-space training implementation passed polynomial-expansion and autograd checks before training. Exact population evaluation is independent polynomial expansion.','',
'These are transfer outcomes at equal source scalar-label budgets, not equal input counts, parameter counts, or computational work. Four held-out128-label tasks add512labels equally to every allocation. TeacherRMSnormalization is a privileged simulation convention. Learning from four output heads need not be uniformly best, and the comparison cannot establish a universally optimal label allocation.','',
'Independent audits replay seeded teacher/input/noise generation, nested budgets, all initializations, direct-forward Jacobians, polynomial population losses,6561-node component matching, polynomial baseline and adaptation normal equations. All unsuccessful and budget-limited fits are retained;32-label adaptation and all source-fit baselines remain in JSON.','',
'Decision: close this fixed confirmation at120fits. Interpret mixed/negative outcomes without altering the recipe. Separate transfer improvement, component recovery and random-initialization robustness before planning any larger-network extension.','']
(root/'FINITE_READOUT_H4_RESULTS.md').write_text('\n'.join(lines))
write(out/'summary.json',dict(status='complete',aggregates=aggregates,comparisons=comparisons,source_sha256=digest(Path(__file__))))
print('\n'.join(lines))
