import hashlib,json
from pathlib import Path
from aggregate_h64 import aggregate

root=Path('results/task_coverage_h64');out=root/'evaluation'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
rows=json.loads((out/'results.json').read_text());audit=json.loads((out/'audit.json').read_text());provenance=json.loads((out/'audit_cases'/'protocol.json').read_text())
assert audit['status']=='passed' and audit['fits']==60
assert provenance['results_sha256']==sha(out/'results.json') and provenance['evaluation_protocol_sha256']==sha(out/'protocol.json')
for name,value in provenance['source_sha256'].items():assert sha(Path(name))==value
checked={r['name']:r for r in audit['records']};assert len(checked)==60
for r in rows:assert checked[r['name']]['input_record_sha256']==hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
summary=aggregate(rows);progress=json.loads((root/'training_progress.json').read_text());groups=progress['groups'];assert len(groups)==20
summary['resources']=[dict(teacher=g['teacher'],n=g['n'],tasks=g['tasks'],**g['resources']) for g in groups]
summary['audit_sha256']=sha(out/'audit.json')
lines=['# d128/h64 task and input scaling','',
'All 60 scientific fits have completed evaluation and audit. Five fresh teachers, two random starts and a privileged teacher-start control per condition. Primary transfer uses 8,192 adaptation labels for 2,080 hidden-product features; 2,048-label diagnostics remain in the raw records.','',
'Errors below are medians across five teacher means, averaging two random starts and four whole-task holdouts within teacher. Teacher starts remain separate. Full recovery requires every matched quadratic-feature cosine to exceed .99. These are finite-budget outcomes after 3,000 Adam steps, not convergence certificates.','',
'| Inputs | Tasks | Initialization | Source MSE | Transfer MSE | All 64 features |',
'|---:|---:|---|---:|---:|---:|']
for c in summary['conditions']:lines.append(f'| {c["n"]} | {c["tasks"]} | {c["initialization"]} | {c["median_source"]:.6g} | {c["median_transfer"]:.6g} | {c["recovery"]}/{c["runs"]} |')
lines+=['','## Paired budget comparisons','',
f'At equal 17,039,360 scalar labels, 520 tasks × 32,768 inputs beats 2,080 tasks × 8,192 inputs on {summary["equal_budget"]["more_inputs_wins"]}/5 teachers. Per-teacher differences, ranges and all conditions are retained in summary.json.']
for n,v in summary['fixed_input'].items():lines.append(f'At {int(n):,} inputs, more tasks improves transfer on {v["more_task_wins"]}/5 teachers; it also adds labels and readout parameters.')
lines+=['','## Optimization witnesses','',
f'{sum(w["lower_by_margin"] for w in summary["lower_loss_witnesses"])}/40 random-start endpoints have empirical loss more than 1e-8 above their same-condition trained teacher-start checkpoint. This is a numerical feasible lower-loss witness, not a certificate of strict local minima or proof that optimization explains all generalization error. Teacher starts are privileged.',
'','## Resources and audit scope','',
f'Twenty independent three-model groups used {sum(g["resources"]["seconds"] for g in groups)/3600:.3f} GPU-training wall hours in total, excluding dataset preparation, serialization and CPU analysis. Maximum torch-allocated memory was {max(g["resources"]["peak_allocated_bytes"] for g in groups)/2**30:.3f} GiB. Do not multiply each group time by its three models.',
'',
'The audit checks all conditions, training/data/code/checkpoint hashes, inactive weights, initial pairing, logged curves, regenerated targets, literal empirical predictions, quadratic-coefficient feature matching and explicit SVD adaptation. Exact population errors use the previously verified Wick kernel, shared between evaluation and audit. It does not replay every optimizer step or independently measure GPU memory. Training is float32 with TF32 off; evaluation is float64.',
'',
'There are 120 learned transfer fits and 10 shared oracle fits. Repeating oracle results across student records yields 240 reported transfer entries, not 240 independent fits. Raw scientific artifacts preserve all starts and controls; no result-based seed selection or optimizer rescue was performed.',
'','## Scope','',
'This is larger-width depth-two shared-feature recovery under known Gaussian normalization. It does not complete depth3–5 recovery, unprivileged graph discovery or prove a general scaling law. Compare the whole table and paired teacher outcomes before choosing a subsequent experiment; the broader research goal remains active.']
Path('TASK_COVERAGE_H64_RESULTS.md').write_text('\n'.join(lines)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('\n'.join(lines))
