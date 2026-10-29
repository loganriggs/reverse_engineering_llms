"""Summarize independently audited depth-2 GPU data-scaling controls."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_svhn_standard import digest, write

parser = argparse.ArgumentParser()
parser.add_argument("--depth", type=int, choices=[2,3,4], default=2)
depth = parser.parse_args().depth
suffix = "" if depth == 2 else f"_DEPTH{depth}"
root = Path(__file__).parent
rows = []
for regime in ['standard', 'extra']:
    for architecture in ['stack', 'tree']:
        out = root / f'results/svhn_gpu/{regime}/d{depth}_{architecture}'
        audit = json.loads((out / 'audit.json').read_text())
        assert audit['status'] == 'passed' and audit['result_sha256'] == digest(out / 'result.json')
        result = json.loads((out / 'result.json').read_text())
        matched = json.loads((out / 'matched_update.json').read_text())
        final = result['curves'][-1]
        rows.append(dict(regime=regime, architecture=architecture, parameters=result['parameters'],
                         matched=matched, final=final))
write(root / ('results/svhn_gpu/summary.json' if depth == 2 else f'results/svhn_gpu/summary_depth{depth}.json'), dict(rows=rows, seeds=1, official_test_used=False))
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for r in rows:
    directory = root / f"results/svhn_gpu/{r['regime']}/d{depth}_{r['architecture']}"
    curves = json.loads((directory / 'result.json').read_text())['curves']
    for ax, metric in zip(axes, ['accuracy', 'ce']):
        ax.plot([c['optimizer_updates'] for c in curves],
                [c['validation'][metric] for c in curves],
                label=f"{r['regime']} / {r['architecture']}")
        ax.scatter([8010], [r['matched']['validation'][metric]], s=15)
for ax, title in zip(axes, ['Validation accuracy', 'Validation cross-entropy']):
    ax.set_xscale('log')
    ax.axvline(8010, color='gray', linestyle=':', linewidth=1)
    ax.set(xlabel='Optimizer updates (log scale)', ylabel=title)
    ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle(f'SVHN depth {depth}: one seed, fixed validation split')
fig.savefig(root / f'figures/svhn_gpu_depth{depth}.png', dpi=160)
plt.close(fig)
lines = ['# SVHN extra-data GPU comparison', '',
    f'Four depth-{depth} fits completed on RTX 5080 in float32 with TF32 disabled. All final training/validation metrics and both final and 8,010-update validation predictions replay independently in NumPy. All models use width 32, per-node RMSNorm, Adam .001, batch 256, gradient clipping at 10, and the same model seed. Equal-width stack/tree functions agree at initialization; tree occurrences have independent parameters.', '',
    'Standard training uses 68,257 examples; added-data training uses 599,388 (including all 531,131 extra examples). Both use the identical 5,000-example validation split and normalization fitted only on standard training. Official test remains unused. No augmentation or convolution.', '',
    '| Data | Architecture | Parameters | Validation at 8,010 updates | Validation at 30 epochs | Final CE | Final train accuracy |',
    '|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['regime']} | {r['architecture']} | {r['parameters']} | {100*r['matched']['validation']['accuracy']:.2f}% | {100*r['final']['validation']['accuracy']:.2f}% | {r['final']['validation']['ce']:.4f} | {100*r['final']['train']['accuracy']:.2f}% |")
lines += ['', f'![Learning curves](figures/svhn_gpu_depth{depth}.png)', '', 'Standard 30 epochs equal 8,010 optimizer updates and 2,047,710 examples seen. Extra-data 30 epochs equal 70,260 updates and 17,981,640 examples seen. Its 8,010-update snapshot sees 2,050,068 examples (slightly more because incomplete epoch-end batches differ). This controls optimizer-update count, not exactly compute, distinct examples, or data distribution.', '',
    'Interpretation: the equal-epoch comparison combines more distinct data with more training. The matched-update comparison constrains that confound but extra images also have a different distribution. This is one initialization on a fixed development split, not a seed-robust advantage or a pure sample-size causal estimate. Training accuracy is measured on different data pools and should not be directly compared as if tasks were identical. Accuracy does not establish circuit recovery.', '',
    'Next: finish paired depth controls and replicate seeds where differences persist. Retain parameter-budget and width-allocation caveats from the previous standard-data studies. Do not infer convergence solely from completing 30 epochs.', '']
(root / f'SVHN_GPU_RESULTS{suffix}.md').write_text('\n'.join(lines))
print('\n'.join(lines))
