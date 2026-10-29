"""Teacher-block summaries of fixed-label-budget noisy recovery, all fits retained."""
import json,statistics
from pathlib import Path
from run_svhn_standard import digest,write
root=Path(__file__).parent;out=root/'results/finite_readout_recovery';audit=json.loads((out/'audit.json').read_text());assert audit['status']=='passed' and len(audit['records'])==72
rows=[]
for a in audit['records']:
 p=out/(a['name']+'.json');assert digest(p)==a['result_sha256'] and digest(p.with_suffix('.npz'))==a['checkpoint_sha256'];rows.append(json.loads(p.read_text()))
lines=['# Finite noisy labels: diversity versus input coverage','','72 fits: three previously used d6/h3teachers, source scalar-label budgets384/1536, one/three/sixheads, normalized label-noise standard deviation0/.05, random or exact-teacher initialization. Each source input carries all selected heads, so n=budget/heads. Input/noise pools and head prefixes are paired within teacher. Same fixedTRFsettings and200function-evaluation cap, no restarts, no validation selection. Teacher starts are privileged basin diagnostics, not a competing practical training method.','',
'Targets use known Gaussian teacher RMS normalization, a privileged simulation convention shared by all arms. The source budget counts scalar labels, not input examples: at384labels the three allocations use384/128/64inputs; at1536they use1536/512/256inputs. Independent Gaussian label noise is added after target normalization. These are finite sample-label experiments, unlike the preceding population coefficient-noise study.','',
'Four whole heads remain held out from representation fitting. Fit linear readouts on frozen hidden products with32or128fresh noisy labels per held-out head; adaptation noise matches the source noise level. At128labels/head this adds512scalar labels equally to every source comparison. True hidden products provide an oracle transfer baseline. A minimum-norm126-monomial quartic regressor provides a source-fit baseline; it has different capacity and is underdetermined in some low-input conditions. All baselines and32-label results are retained in JSON.','',
'## Noisy-label transfer (standard deviation .05)','','Primary displayed endpoint: mean exact Gaussian MSE over the four held-out tasks after128adaptation labels/head. Table shows median across three teachers, followed by the full range. Counts of samples/parameters differ by allocation; this is a resource-allocation comparison, not pure causal isolation of head count.','',
'| Source labels | Initialization | Heads | Median held-out MSE (range) | All component cosines>.99 | Budget-limited fits |','|---|---|---:|---:|---:|---:|']
for budget in [384,1536]:
 for init in ['random','teacher']:
  for m in [1,3,6]:
   a=[r for r in rows if r['budget']==budget and r['initialization']==init and r['heads']==m and r['noise_std']==.05];values=[statistics.mean(r['transfer_mse']['learned_128']) for r in a]
   lines.append(f"| {budget} | {init} | {m} | {statistics.median(values):.6g} ({min(values):.6g}–{max(values):.6g}) | {sum(min(r['component_cosines'])>.99 for r in a)}/3 | {sum(r['status']==0 for r in a)}/3 |")
lines+=['','The oracle128-label transfer errors are .0001980/.0003739/.0004108 for teachers218001/2/3, identical across source allocations and initializations. Poorer learned-feature transfer can therefore persist even when the adaptation procedure itself works with the correct features.','',
'## Every paired noisy transfer result','','| Source labels | Initialization | Teacher | One head | Three heads | Six heads |','|---|---|---|---:|---:|---:|']
for budget in [384,1536]:
 for init in ['random','teacher']:
  for seed in [218001,218002,218003]:
   a={r['heads']:r for r in rows if r['budget']==budget and r['initialization']==init and r['teacher_seed']==seed and r['noise_std']==.05}
   lines.append(f'| {budget} | {init} | {seed} | '+' | '.join(f"{statistics.mean(a[m]['transfer_mse']['learned_128']):.6g}" for m in [1,3,6])+' |')
lines+=['','## Noiseless control','','| Source labels | Random-start heads | All component cosines>1−1e-6 |','|---|---:|---:|']
for budget in [384,1536]:
 for m in [1,3,6]:
  a=[r for r in rows if r['budget']==budget and r['initialization']=='random' and r['heads']==m and r['noise_std']==0]
  lines.append(f"| {budget} | {m} | {sum(min(r['component_cosines'])>1-1e-6 for r in a)}/3 |")
lines+=['','All noiseless teacher starts retain the exact solution, as expected; this is an implementation/control check, not learning success.','',
'## Interpretation and limits','',
'Three heads beat one on noisy held-out transfer in3/3random-start cases at384labels and2/3at1536. Teacher-initialized comparisons are2/3and3/3respectively. More heads are not uniformly better: six heads sacrifice input coverage and introduce additional readout parameters, and random-start optimization failures remain. The1536-label three-head random-start failure on teacher218003 has transferMSE.0384versus.000685from the teacher start; median-only reporting would obscure it.','',
'The preceding additive-information argument held inputs/noise conventions fixed while adding heads. It does not imply monotonic improvement when a fixed label budget forces fewer distinct inputs. This panel demonstrates why annotation allocation and optimization must be measured separately. Only three teachers and one random initialization per allocation are used; the teachers were used in earlier development experiments. No broad statistical claim or chosen optimal head count follows.','',
'Independent audits replay seeded input/noise generation, source/held-out separation, total budgets, direct-forward Jacobians, polynomial population losses,729-node component comparisons, polynomial baseline normal equations and all transfer solves. All72fits retained; maximum relative Jacobian error1.43e-15 and training-MSE discrepancy3.25e-15. Termination at a local nonzero residual is retained even when the optimizer reports success.','',
'Next: preregister a fresh-teacher confirmation of the fixed-budget comparison, using additional random initializations and teacher-level aggregation. Keep three heads as a candidate rather than selecting it as a universal optimum. Larger-scale or new-task claims should wait for this replication.','']
(root/'FINITE_READOUT_RECOVERY_RESULTS.md').write_text('\n'.join(lines))
write(out/'summary.json',dict(status='complete',rows=rows,source_sha256=digest(Path(__file__))))
print('\n'.join(lines))
