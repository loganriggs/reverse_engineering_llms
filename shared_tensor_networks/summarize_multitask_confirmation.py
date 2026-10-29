import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).parent;out=root/'results/multitask_confirmation'
    rows=list(csv.DictReader((out/'transfer.csv').open()));fits=list(csv.DictReader((out/'fits.csv').open()))
    summary=[];source=[]
    for kind in ['shared','independent']:
        for n in [256]:
            for ts in range(5):
                for method in ['shared','shared_wide','separate_width','separate_budget']:
                    group=[r for r in fits if r['teacher']==kind and int(r['n'])==n and int(r['teacher_seed'])==ts and r['model']==method]
                    source.append(dict(teacher=kind,n=n,teacher_seed=ts,method=method,mean_population_mse=statistics.mean(float(r['population_mse']) for r in group)))
            for k in [8,32,128]:
                for method in sorted({r['method'] for r in rows}):
                    for solver in ['minnorm','ridge_fixed']:
                        for ts in range(5):
                            group=[r for r in rows if r['teacher']==kind and (int(r['n'])==n or method=='oracle_task_outputs') and int(r['k'])==k and int(r['teacher_seed'])==ts and r['method']==method and r['solver']==solver]
                            summary.append(dict(teacher=kind,n=n,k=k,method=method,solver=solver,teacher_seed=ts,
                                                mean_population_mse=statistics.mean(float(r['population_mse']) for r in group),
                                                mean_ood_mse=statistics.mean(float(r['ood_mse']) for r in group)))
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,squeeze=False)
    for col,kind in enumerate(['shared','independent']):
        for row,n in enumerate([256]):
            ax=axes[row,col]
            for method,label in [('shared','Shared'),('shared_wide','Shared, wider'),('separate_width','Separate, same width'),('separate_budget','Separate, near budget'),('polynomial','Polynomial baseline'),('oracle_task_outputs','True source-task outputs')]:
                blocks=[[r['mean_population_mse'] for r in summary if r['teacher']==kind and r['n']==n and r['k']==k and r['method']==method and r['solver']=='ridge_fixed'] for k in [8,32,128]]
                line,=ax.plot([8,32,128],[statistics.median(b) for b in blocks],'o-',label=label)
                ax.fill_between([8,32,128],[min(b) for b in blocks],[max(b) for b in blocks],color=line.get_color(),alpha=.1)
            ax.axhline(1,color='black',linestyle=':',linewidth=1)
            ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks([8,32,128],labels=['8','32','128'])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator());ax.grid(alpha=.2)
            ax.set_title(f'{kind} teacher; {n} source examples/task');ax.set_xlabel('Labels per held-out task');ax.set_ylabel('Population MSE')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Fixed ridge: median and full range of five teacher-level means\nEach teacher mean pools two initializations and four held-out tasks; bands are not confidence intervals')
    fig.tight_layout();fig.savefig(root/'figures/multitask_confirmation.png',dpi=150)
    (out/'summary.json').write_text(json.dumps(dict(transfer_teacher_blocks=summary,source_teacher_blocks=source),indent=2)+'\n')
    for kind in ['shared','independent']:
        print(kind,'N256 source teacher-block medians',{m:statistics.median(r['mean_population_mse'] for r in source if r['teacher']==kind and r['n']==256 and r['method']==m) for m in ['shared','shared_wide','separate_width','separate_budget']})


if __name__=='__main__':main()
