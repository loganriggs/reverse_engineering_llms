"""Fresh-seed replication summary; original development seed kept separate."""
import hashlib,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path(__file__).parent;src=root/'results/fashion_replication';seeds=[190001,190002,190003];records=[]
    for seed in seeds:
        folder=src/f'seed{seed}';audit=json.loads((folder/'audit.json').read_text());assert audit['status']=='passed' and len(audit['rows'])==10
        for a in audit['rows']:
            r=json.loads((folder/f"{a['name']}.json").read_text());assert len(r['curves'])==30
            records.append(dict(seed=seed,name=r['name'],parameters=r['parameters'],depth=r['depth'],architecture=r['architecture'],train=r['curves'][-1]['train'],validation=r['curves'][-1]['validation'],input_sha256=hashlib.sha256((folder/f"{a['name']}.json").read_bytes()).hexdigest()))
    text='''# Fresh-seed full-pool Fashion-MNIST replication

Three fresh model initialization seeds,55,000training/5,000validation,30epochs, width32, RMSNorm depths2–4, stack/equal-width tree/near-budget narrow tree andReLUbaseline. Same fixed recipe and minibatch orders as the earlier full-pool run; no early stopping or best-epoch selection. This isolates initialization variation on one fixed data split, not uncertainty across independent datasets. The original development seed is not pooled into these fresh-seed summaries. Official test remains unused.

All30models complete; independent NumPy train/validation/logit replays pass. CPUonly, genericdispatch with each producer pinned to onecore. Code, protocols, final states, optimizer states and full curves retained.

| Model | Parameters | Mean train accuracy | Mean validation accuracy | Validation range across3seeds | Mean validation CE |
|---|---:|---:|---:|---:|---:|
'''
    names=[f'd{d}_{k}' for d in [2,3,4] for k in ['stack','tree','tree_budget']]+['d0_mlp'];summary=[]
    for name in names:
        rr=[r for r in records if r['name']==name];acc=[r['validation']['accuracy'] for r in rr];item=dict(name=name,parameters=rr[0]['parameters'],mean_validation_accuracy=statistics.mean(acc),min_validation_accuracy=min(acc),max_validation_accuracy=max(acc),mean_train_accuracy=statistics.mean(r['train']['accuracy'] for r in rr),mean_validation_ce=statistics.mean(r['validation']['ce'] for r in rr));summary.append(item)
        text+=f"| {name} | {item['parameters']} | {item['mean_train_accuracy']:.2%} | {item['mean_validation_accuracy']:.2%} | {min(acc):.2%}–{max(acc):.2%} | {item['mean_validation_ce']:.4f} |\n"
    text+='\n## Paired validation-accuracy differences\n\nDifferences below are percentage points versus the stack at the same depth and seed. Three seeds do not justify a universal architecture ranking. Equal-width trees have more parameters; budget trees change width.\n\n'
    paired=[]
    for d in [2,3,4]:
        for kind in ['tree','tree_budget']:
            delta=[]
            for seed in seeds:
                s=next(r for r in records if r['seed']==seed and r['name']==f'd{d}_stack');t=next(r for r in records if r['seed']==seed and r['name']==f'd{d}_{kind}');delta.append(100*(t['validation']['accuracy']-s['validation']['accuracy']))
            paired.append(dict(depth=d,architecture=kind,delta_percentage_points=delta));text+=f"- Depth{d} {kind}: {', '.join(f'{v:+.2f}' for v in delta)} points; mean {statistics.mean(delta):+.2f}.\n"
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(range(len(summary)),[100*r['mean_validation_accuracy'] for r in summary],color=['#4c78a8','#f58518','#54a24b']*3+['#888888']);ax.errorbar(range(len(summary)),[100*r['mean_validation_accuracy'] for r in summary],yerr=[[100*(r['mean_validation_accuracy']-r['min_validation_accuracy']) for r in summary],[100*(r['max_validation_accuracy']-r['mean_validation_accuracy']) for r in summary]],fmt='none',color='black',capsize=3);ax.set_xticks(range(len(summary)),[r['name'] for r in summary],rotation=35,ha='right');ax.set_ylabel('Validation accuracy (%)');ax.set_title('Fresh3seed means and ranges (not confidence intervals)');ax.set_ylim(0,100);fig.tight_layout();fig.savefig(root/'figures/fashion_replication.png',dpi=150);plt.close(fig)
    text+='\nEqual-width trees gain mean accuracy by about0.47/0.63/0.58percentagepoints atdepth2/3/4, but mean validationCE is worse at everydepth. This is a small accuracy improvement with more parameters, not an across-the-board benefit. Budget trees are approximately comparable atdepth2 and weaker atdepth3/4; thedepth4budget condition is notably variable. Initializations share the same data split, and three seeds do not establish generality.\n'
    text+='\n![Fresh seed comparison](figures/fashion_replication.png)\n\nNext: broader SVHN standard/extra training comparison after checksum verification, and sample-size controls. These classification results do not establish reusable circuit recovery. Keep parameter tying, cached computation and function complexity distinct.\n'
    (root/'FASHION_REPLICATION_RESULTS.md').write_text(text);(src/'summary.json').write_text(json.dumps(dict(records=records,summary=summary,paired=paired),indent=2)+'\n');print(json.dumps(paired))
if __name__=='__main__':main()
