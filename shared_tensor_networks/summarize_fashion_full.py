"""Summarize audited full-data curves without best-validation selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path(__file__).parent;out=root/'results/fashion_full';audit=json.loads((out/'audit.json').read_text());assert audit['status']=='passed'
    names=[r['name'] for r in audit['rows']];rows=[json.loads((out/f'{n}.json').read_text()) for n in names]
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for d in [2,3,4]:
        for r in rows:
            if r['depth']!=d:continue
            c=r['curves'];xs=[v['epoch'] for v in c]
            axes[0,d-2].plot(xs,[v['validation']['accuracy'] for v in c],label=r['architecture'])
            line=axes[1,d-2].plot(xs,[v['validation']['ce'] for v in c],label=r['architecture'])[0]
            axes[1,d-2].plot(xs,[v['train']['ce'] for v in c],ls='--',color=line.get_color())
        axes[0,d-2].set_title(f'{d} bilinear layers + RMSNorm');axes[0,d-2].legend();axes[1,d-2].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Validation accuracy');axes[1,0].set_ylabel('CE: validation solid, train dashed');fig.suptitle('55,000 training / 5,000 development examples; one seed; official test unused');fig.tight_layout();fig.savefig(root/'figures/fashion_full.png',dpi=150);plt.close(fig)
    text='''# Full-pool Fashion-MNIST development experiment

Ten classifiers trained on55,000 examples for30epochs, with5,000 validation examples and untouched official test. One seed, fixed30epoch results. RMSNorm at all bilinear nodes; no augmentation, no weight decay. Adam lr.001, batch256, clipping10. Same initial function for equal-width stack/tree at each depth. Near-budget trees change width; this comparison does not isolate parameter count from geometry. ReLU baseline has two64-wide hidden layers.

All10 final checkpoints replayed independently in NumPy, including train and validation cross-entropy/accuracy, saved validation logits, counts, split identities and source hashes. Repeated native SIGILL/SIGSEGV crashes interrupted depth4 budget-tree training. Generic CPU dispatch alone also crashed. The remaining run completed with ATEN_CPU_CAPABILITY=default and CPU affinity pinned to core0, resuming saved model/optimizer checkpoints and deterministic epoch minibatch order. Crash traces and runtime changes retained; a successful pinned run does not prove the root cause or fix. Runtime reliability remains an open issue; this is not evidence of architectural numerical failure.

| Model | Parameters | Train accuracy | Validation accuracy | Validation CE | CE change: epochs26–30 minus21–25 |
|---|---:|---:|---:|---:|---:|
'''
    for r in rows:
        c=r['curves'];last=c[-1];delta=np.mean([v['validation']['ce'] for v in c[-5:]])-np.mean([v['validation']['ce'] for v in c[-10:-5]])
        text+=f"| {r['name']} | {r['parameters']} | {last['train']['accuracy']:.2%} | {last['validation']['accuracy']:.2%} | {last['validation']['ce']:.4f} | {delta:+.4f} |\n"
    text+='''
![Learning curves](figures/fashion_full.png)

These curves separate fitting from generalization. Large increases over the small pilot show that short low-data fits could not establish architecture limits. Equal-width trees generally fit training data more closely, but validation differences are small and cannot support a ranking from one seed. Late validation CE changes are descriptive, not statistical convergence tests. Do not call this a sample-complexity study: data volume, number of updates and training-fitted normalization all changed from the pilot.

Next: seed replication and nested training-size curves with both epoch- and update-budget accounting. Inspect whether longer training overfits before indiscriminately extending. Add SVHN standard versus standard-plus-extra data with separate provenance. No official-test tuning. Do not claim circuit recovery from image accuracy. Theory still concerns sharing separately from repeated inputs and physical execution reuse; normalization removes the polynomial model assumption.
'''
    (root/'FASHION_FULL_RESULTS.md').write_text(text)
if __name__=='__main__':main()
