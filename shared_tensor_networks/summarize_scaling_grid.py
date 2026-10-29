"""Teacher-level summaries; tasks and evaluation batches are not replicates."""
import json,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).parent;out=root/'results/scaling_grid';raw=json.loads((out/'results.json').read_text());assert raw['status']=='complete'
    audit=json.loads((out/'audit.json').read_text());assert audit['status']=='passed'
    rows=[r for r in raw['rows'] if not r.get('failure')];summary=[];paired=[]
    fig,axes=plt.subplots(2,4,figsize=(16,7),sharey=True)
    for gi in range(4):
        for ki,kind in enumerate(['shared','independent']):
            group=[r for r in rows if r['grid_index']==gi and r['teacher']==kind];ax=axes[ki,gi]
            for mi,model in enumerate(['shared','separate','shared_budget']):
                selected=sorted([r for r in group if r['model']==model],key=lambda r:r['replicate'])
                if not selected:continue
                values=np.array([np.array(r['transfer_relative_mse']).mean(axis=1) for r in selected]);source=[r['source_relative_mse'] for r in selected]
                gates=[]
                for r in selected:
                    cv=np.array(r['metric_batch_cv']);gates.extend(((cv[2:10]<.1)&(cv[18:26]<.1)).tolist())
                record=dict(grid_index=gi,teacher=kind,model=model,replicates=len(selected),median_source_relative_mse=statistics.median(source),median_transfer_and_oracle= np.median(values,axis=0).tolist(),source_precision_passes=sum(r['source_precision'] for r in selected),learned_transfer_task_precision_passes=sum(gates),learned_transfer_task_metrics=len(gates))
                summary.append(record)
                ax.scatter(np.full(len(selected),mi),values[:,1],s=22)
                ax.plot([mi-.15,mi+.15],[np.median(values[:,1])]*2,color='black')
            for rep in range(3):
                chosen=[next((r for r in group if r['replicate']==rep and r['model']==m),None) for m in ['shared','separate','shared_budget']]
                if all(r is not None for r in chosen):ax.plot(range(3),[np.mean(r['transfer_relative_mse'][1]) for r in chosen],color='gray',alpha=.25)
            a=sorted([r for r in group if r['model']=='shared_budget'],key=lambda r:r['replicate']);b=sorted([r for r in group if r['model']=='separate'],key=lambda r:r['replicate'])
            if len(a)==len(b)==3:
                paired.append(dict(grid_index=gi,teacher=kind,budget_vs_separate_wins={str(k):sum(bool(np.mean(x['transfer_relative_mse'][j])<np.mean(y['transfer_relative_mse'][j])) for x,y in zip(a,b)) for j,k in enumerate([32,128])}))
            ax.axhline(1,ls=':',color='black',lw=1);ax.set_yscale('log');ax.set_xticks(range(3),['Shared','Separate','Shared\nnear budget']);ax.grid(alpha=.15)
            d,h,depth=[(16,8,2),(16,8,3),(32,16,2),(32,16,3)][gi];ax.set_title(f'd{d}, h{h}, depth{depth}; {kind} teacher')
            if gi==0:ax.set_ylabel('Relative heldout-task error (128 labels)')
    fig.suptitle('Three teacher replicates; points average four whole-task holdouts\nGray lines connect teachers; black ticks are medians. Precision failures are retained.')
    fig.tight_layout();fig.savefig(root/'figures/scaling_grid.png',dpi=150)
    (out/'summary.json').write_text(json.dumps(dict(groups=summary,paired=paired,failures=[r for r in raw['rows'] if r.get('failure')]),indent=2)+'\n')
    for row in summary:print(row['grid_index'],row['teacher'],row['model'],round(row['median_source_relative_mse'],4),[round(v,4) for v in row['median_transfer_and_oracle']],row['source_precision_passes'])
    print('Paired',paired)


if __name__=='__main__':main()
