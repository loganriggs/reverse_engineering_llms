"""Empirical versus population extraction metrics, preserving every teacher/model."""
import json,statistics
from pathlib import Path
from run_svhn_standard import digest,write
root=Path(__file__).parent;out=root/'results/empirical_extraction';audit=json.loads((out/'audit.json').read_text());assert audit['status']=='passed' and len(audit['records'])==300
rows=[]
for a in audit['records']:
 p=out/(a['name']+'.json');assert digest(p)==a['result_sha256'] and digest(p.with_suffix('.npz'))==a['checkpoint_sha256'];rows.append(json.loads(p.read_text()))
base=json.loads((root/'results/hidden_extraction/summary.json').read_text())['aggregates'];aggs=[]
for method in ['shared','separate_width','released','teacher']:
 for extraction in ['quadratic_consensus','quartic_pca']:
  for pool in ['source512','fresh512','fresh4096']:
   for seed in range(225001,225006):
    a=[r for r in rows if r['method']==method and r['extraction']==extraction and r['pool']==pool and r['teacher_seed']==seed];assert len(a)==(1 if method=='teacher' else 3)
    def metric(r,label):return statistics.mean(next(m['mse'] for m in r['metrics'] if m['labels']==label))
    aggs.append(dict(method=method,extraction=extraction,pool=pool,seed=seed,transfer=statistics.mean(metric(r,128) for r in a),floor=statistics.mean(metric(r,'population') for r in a)))
lines=['# Empirical activation metrics for hidden extraction','','300fixed extractions from50stored models, each with population projection and32/128label adaptation:900fits independently audited. No source retraining or new target labels. Same fixed rank4quadratic→10products and rank10quartic-union decompositions as the previous panel, now using activation column norms and SVDs estimated on512source inputs,512fresh unlabeled inputs, or4096fresh unlabeled inputs. Fresh pools are nested and fixed per teacher.','',
'Analytic Gaussian moments do not choose the empirical features; they are used only to evaluate final population error. Source512adds neither labels nor input observations. Fresh pools add unlabeled observations. Ranks remain a known architecture budget rather than an automatically discovered manifold dimension. All original failed fits are retained.','',
'## Quadratic consensus: median teacher transfer MSE','','| Model origin | Analytic metric | Source512 | Fresh512 | Fresh4096 |','|---|---:|---:|---:|---:|']
for method in ['shared','separate_width','released','teacher']:
 values=[statistics.median(r['transfer'] for r in base if r['method']==method and r['extraction']=='quadratic_consensus')]
 for pool in ['source512','fresh512','fresh4096']:values.append(statistics.median(r['transfer'] for r in aggs if r['method']==method and r['extraction']=='quadratic_consensus' and r['pool']==pool))
 lines.append('| '+method+' | '+' | '.join(f'{v:.6g}' for v in values)+' |')
lines+=['','Transfer averages four whole held-out tasks at128noisy adaptation labels/head, then all original initialization outcomes within each teacher. Five teacher aggregates form the median; these are reused models, not fresh-teacher confirmation.','',
'## Ranges matter','','| Model origin | Extraction | Metric pool | Median transfer | Full teacher range |','|---|---|---|---:|---:|']
for method in ['released','teacher']:
 for extraction in ['quadratic_consensus','quartic_pca']:
  for pool in ['source512','fresh512','fresh4096']:
   values=[r['transfer'] for r in aggs if r['method']==method and r['extraction']==extraction and r['pool']==pool]
   lines.append(f'| {method} | {extraction} | {pool} | {statistics.median(values):.6g} | {min(values):.6g}–{max(values):.6g} |')
lines+=['','The main compression result survives without an analytic metric: released-model quadratic consensus is about.00121using existing source inputs, versus.03436for their original source outputs. It does not repair all failed shared starts. Random-start independent models remain poor under every metric pool. Increasing unlabeled sample count does not monotonically improve transfer for these particular draws. All empirical and analytic methods agree for shared models because they preserve the same full feature span.','',
'## Why higher-degree metric estimates can be noisy','',
'For a Gaussian monomial z=product_i X_i^a_i with independent standard-normal coordinates, an iid empirical second moment has relative variance','',
'Var(mean(z²))/E[z²]² = (product_i E[X_i^(4a_i)] / product_i E[X_i^(2a_i)]² − 1)/M.','',
'This follows directly from variance of an iid sample mean and factorization of Gaussian moments; E[X^(2k)]=(2k−1)!!. It concerns second-moment estimates, not the standard deviation of estimatedRMSnorms or a full PCA-error theorem.','',
'| Feature | Relative variance × M | Relative SD at512inputs | Relative SD at4096inputs |','|---|---:|---:|---:|']
checks=json.loads((out/'metric_variance_checks.json').read_text());assert checks['status']=='passed'
for r in checks['rows']:lines.append(f"| {r['name']} | {r['relative_variance_times_sample_count']} | {r['relative_std_second_moment_512']:.3f} | {r['relative_std_second_moment_4096']:.3f} |")
lines+=['','For example, a product of four independent Gaussian coordinates has ten times the relative second-moment variance of a product of two. Squared bilinear features can be noisier still. Exact rational moment calculations agree with9-point Gaussian quadrature. These examples motivate caution with quartic empirical metrics; they do not prove that degree alone determines stability for the correlated learned features or that quadratic extraction universally wins.','',
'## Verification and decision','',
'Independent audit recomputes empirical norms from direct hidden activations, verifies leading right-singular eigenspaces and coefficient reconstruction, checks empirical product normalization and forward values, then replays every adaptation normal equation and exact Gaussian error. All300extractions/900evaluations pass. No teacher feature directions or held-out labels enter rank/subspace selection.','',
'This removes the analytic-metric assumption for the observed good-basin compression result. It does not remove the fixed-rank prior, Gaussian task setting, small network size or random-start failures. Further work should target those limitations or synthesize the theory/evidence, rather than simply repeating more empirical pool sizes.','']
(root/'EMPIRICAL_EXTRACTION_RESULTS.md').write_text('\n'.join(lines));write(out/'summary.json',dict(status='complete',aggregates=aggs,source_sha256=digest(Path(__file__))))
print('\n'.join(lines))
