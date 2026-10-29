"""Require all30 independent audits, then report paired data/capacity comparisons."""
import json
from pathlib import Path
import numpy as np
from run_svhn_standard import digest,write

root=Path(__file__).parent
out=root/'results/svhn_confirmation'
records=[]
for seed in [212001,212002,212003,212004,212005]:
    for regime in ['standard','extra']:
        for architecture in ['stack','tree','wide_stack']:
            path=out/f'seed{seed}/{regime}/d3_{architecture}'
            audit=json.loads((path/'audit.json').read_text())
            assert audit['status']=='passed' and audit['result_sha256']==digest(path/'result.json')
            for item in audit['records']:
                assert digest(path/item['checkpoint'])==item['sha256']
            result=json.loads((path/'result.json').read_text())
            records.append(dict(seed=seed,regime=regime,architecture=architecture,parameters=result['parameters'],
                final=result['curves'][-1],matched=json.loads((path/'matched_update.json').read_text()),
                initial_statistics=json.loads((path/'initial_statistics.json').read_text())))
groups=[]
for regime in ['standard','extra']:
    for architecture in ['stack','tree','wide_stack']:
        rows=[r for r in records if r['regime']==regime and r['architecture']==architecture]
        groups.append(dict(regime=regime,architecture=architecture,parameters=rows[0]['parameters'],
            final_accuracy=[r['final']['validation']['accuracy'] for r in rows],
            final_ce=[r['final']['validation']['ce'] for r in rows],
            matched_accuracy=[r['matched']['validation']['accuracy'] for r in rows]))
paired=[]
for seed in [212001,212002,212003,212004,212005]:
    lookup={(r['regime'],r['architecture']):r for r in records if r['seed']==seed}
    item=dict(seed=seed)
    for point in ['final','matched']:
        for regime in ['standard','extra']:
            for comparison in ['stack','wide_stack']:
                item[f'{point}_{regime}_tree_minus_{comparison}_accuracy_pp']=100*(lookup[regime,'tree'][point]['validation']['accuracy']-lookup[regime,comparison][point]['validation']['accuracy'])
        item[f'{point}_data_architecture_interaction_pp']=item[f'{point}_extra_tree_minus_stack_accuracy_pp']-item[f'{point}_standard_tree_minus_stack_accuracy_pp']
    paired.append(item)
write(out/'summary.json',dict(status='complete',records=records,groups=groups,paired=paired,source_sha256=digest(Path(__file__))))
lines=['# Five-seed SVHN data and capacity confirmation','','All30fits and independent audits complete. Five fresh model seeds212001–212005, one fixed training/validation split and minibatch-order schedule. The previous development seed is not pooled. Three depth3 architectures: width32 shared stack, width32 occurrence tree, and width120 shared stack selected solely to approximate tree parameter count. All use identical Adam/RMSNorm/30epoch settings; checkpoint at8010updates retained. No official test use, best-epoch selection, augmentation, or convolution.','',
    '| Data | Model | Parameters | Mean final accuracy (range) | Mean final CE | Mean8010-update accuracy |',
    '|---|---|---:|---:|---:|---:|']
for g in groups:
    a=np.array(g['final_accuracy'])*100
    lines.append(f"| {g['regime']} | {g['architecture']} | {g['parameters']} | {a.mean():.2f}% ({a.min():.2f}–{a.max():.2f}) | {np.mean(g['final_ce']):.4f} | {100*np.mean(g['matched_accuracy']):.2f}% |")
lines+=['','## Paired differences for every seed','','| Seed | Standard tree−stack | Extra tree−stack | Extra tree−wide stack | Data×architecture interaction |','|---|---:|---:|---:|---:|']
for r in paired:
    keys=['final_standard_tree_minus_stack_accuracy_pp','final_extra_tree_minus_stack_accuracy_pp','final_extra_tree_minus_wide_stack_accuracy_pp','final_data_architecture_interaction_pp']
    lines.append('| '+str(r['seed'])+' | '+' | '.join(f'{r[k]:+.2f}' for k in keys)+' |')
lines+=['','All differences above are percentage points, with final accuracy as the endpoint. The JSON also retains matched-update comparisons, cross-entropies, full per-seed metrics and initialization output/gradient statistics.','',
    '## What the confirmation changes','',
    'With extra data, the tree beats the narrow shared stack in accuracy in all five seeds (mean +1.70 percentage points), but the parameter-matched wide shared stack beats the tree in all five (mean +0.78 points). The wide stack also beats the tree at the matched 8010-update checkpoint in all five seeds (mean +1.07 points). This does not support an intrinsic accuracy advantage from untying: allocating a comparable parameter budget to shared width is competitive and better here. Width and initial functions differ, so this is a resource-allocation comparison, not a pure causal estimate of tying.','',
    'The metric choice matters: the extra-data tree has lower cross-entropy than the wide stack in all five seeds (mean .4089 versus .4874), despite lower accuracy. Cross-entropy also depends on confidence and the severity of mistakes; these numbers alone do not identify miscalibration as the cause. Standard-data tree versus narrow-stack accuracy is positive in four of five seeds; the data-by-architecture interaction is positive in four of five and averages only +.57 points. The earlier single-seed ranking reversal is therefore not a robust description of this panel.','',
    'Decision: close this prespecified panel without adding selected seeds or changing hyperparameters. Keep both accuracy and cross-entropy in future optimizer comparisons. The next learning-dynamics experiment should use paired synthetic initial functions and explicit work budgets to separate conditioning from architecture allocation; image accuracy does not establish recovery of reusable computations.','',
    'Interpretation limits: the equal-width tree has more parameters; the wide stack controls approximate parameter count but changes width, initial function and possible function class. Standard30epochs has8010updates, extra30epochs70260updates. Extra data changes distribution as well as example count. Five initialization seeds on one split do not establish dataset-level generality or a confidence interval for new tasks. No circuit recovery is inferred from classification accuracy.','',
    'Independent checks replay final full-training/validation metrics and checkpoint validation logits using NumPy; verify split/data hashes, parameter budgets, seeded initialization statistics, and equal-width initial functions. Any runtime failures remain in logs and panel status records rather than being treated as optimization failures.','']
(root/'SVHN_CONFIRMATION_RESULTS.md').write_text('\n'.join(lines))
print('\n'.join(lines))
