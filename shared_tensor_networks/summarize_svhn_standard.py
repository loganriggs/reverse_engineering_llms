"""Final-epoch summary of audited standard-only SVHN classifiers."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path(__file__).parent;src=root/'results/svhn_standard';rows=[]
    for d in [2,3,4]:
        folder=src/f'depth{d}';a=json.loads((folder/'audit.json').read_text());assert a['status']=='passed'
        for record in a['records']:
            p=folder/f"{record['name']}.json";assert hashlib.sha256(p.read_bytes()).hexdigest()==record['result_sha256'];rows.append(json.loads(p.read_text()))
    assert len(rows)==10
    text='''# Standard-only SVHN: full-pool development comparison

All10classifiers trained for30epochs on68,257standard-training images;5,000validation images from the same official training split. Official test untouched. One initialization seed, shared data/minibatch order, Adam.001,batch256,clip10. RGB channel statistics fit on training only, flatten NCHW, no augmentation. RMSNorm at every bilinear node. Finalepoch results, no best-epoch selection or early stopping. This is a new dataset development experiment, not a multi-seed benchmark claim.

Raw data from UCSDmirror matched the reference archive checksum; cachedarray SHA256 and splits verified. Independent integer-channel-histogram moments verify normalization. NumPy forwards independently replay train/validation metrics and saved logits; matched initial stack/tree functions checked. All training isCPU. A separate brief RTX5080agreement test passed: maximum relative float32gradient difference~2.92e-6 andfloat64~4e-15, withTF32disabled. NoGPUtraining sweep has run.

| Model | Parameters | Dense MAC/example | Train accuracy | Validation accuracy | Validation CE |
|---|---:|---:|---:|---:|---:|
'''
    for r in rows:
        last=r['curves'][-1];text+=f"| {r['name']} | {r['parameters']} | {r['dense_multiply_accumulates_per_example']} | {last['train']['accuracy']:.2%} | {last['validation']['accuracy']:.2%} | {last['validation']['ce']:.4f} |\n"
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for d in [2,3,4]:
        for r in rows:
            if r['depth']!=d:continue
            c=r['curves'];x=[v['epoch'] for v in c];axes[0,d-2].plot(x,[v['validation']['accuracy'] for v in c],label=r['architecture']);line=axes[1,d-2].plot(x,[v['validation']['ce'] for v in c],label=r['architecture'])[0];axes[1,d-2].plot(x,[v['train']['ce'] for v in c],ls='--',color=line.get_color())
        axes[0,d-2].set_title(f'{d} bilinear layers + RMSNorm');axes[0,d-2].legend();axes[1,d-2].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Validation accuracy');axes[1,0].set_ylabel('CE: validation solid, train dashed');fig.suptitle('SVHN standard only: 68,257 train / 5,000 validation; one seed');fig.tight_layout();fig.savefig(root/'figures/svhn_standard.png',dpi=150);plt.close(fig)
    text+='''
The depth2tree is essentially tied with the stack (84.92% versus84.86%). Atdepth3/4the equalwidthtree fits training better but validates worse, especiallydepth4(80.16% versus84.58%). The depth4uniformbudgettree reaches only65.78% validation and66.68% training, consistent with a fitting bottleneck under this recipe. One seed doesnotestablish an architecture law. A proposed budget-allocation control keeps upperlayerswidth32 while reducing leafwidth, to separate totalparameterbudget from a narrowed outputrepresentation.

Dense MAC counts include matrix products only, excluding pointwise products, normalization, activations and memory costs. They are not measured FLOPs or wall-time guarantees. Near-budget trees change width; equal-width trees change parameter counts and execution cost. Hold those distinctions when interpreting accuracy.

![Learning curves](figures/svhn_standard.png)

Next: standard-plus-extra data with the same standard validation split and normalization, and a matched-update checkpoint/control as well as equal-epoch results. Do not equate extra-data gains with a causal sample-size effect without accounting for optimizerupdates and theextra split's distribution. Replicate seeds before general architecture claims. Classification accuracy does not identify reusable circuits.
'''
    (root/'SVHN_STANDARD_RESULTS.md').write_text(text);(src/'summary.json').write_text(json.dumps(dict(rows=[{k:v for k,v in r.items() if k!='curves'}|dict(final=r['curves'][-1]) for r in rows]),indent=2)+'\n');print('Summarized10auditedfits')
if __name__=='__main__':main()
