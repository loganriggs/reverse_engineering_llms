import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).parent;rows=list(csv.DictReader((root/'results/longer_training/results.csv').open()))
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True);summary=[]
    for col,teacher in enumerate(['tied','untied']):
        for row,n in enumerate([256,1024]):
            ax=axes[row,col]
            for student in ['tied','untied_width','untied_budget']:
                values=[];floors=[]
                for step in [600,1800,6000]:
                    group=[r for r in rows if r['teacher']==teacher and int(r['n'])==n and r['student']==student and int(r['step'])==step]
                    value=statistics.median(float(r['exact_gaussian_mse']) for r in group)
                    floor=statistics.median(float(r['oracle_hidden_population_floor']) for r in group)
                    values.append(value);floors.append(floor)
                    summary.append(dict(teacher=teacher,n=n,student=student,step=step,median_mse=value,median_hidden_floor=floor))
                line,=ax.plot([600,1800,6000],values,'o-',label=student)
                ax.plot([600,1800,6000],floors,'--',color=line.get_color(),alpha=.7)
            ax.set_title(f'{teacher} teacher, N={n}');ax.set_xscale('log');ax.set_yscale('log');ax.grid(alpha=.2)
            ax.set_xticks([600,1800,6000],labels=['600','1,800','6,000']);ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlabel('Adam steps');ax.set_ylabel('Exact population MSE')
    axes[0,0].legend(fontsize=8);fig.suptitle('Five-seed medians: solid = model error; dashed = hidden-span floor')
    fig.tight_layout();fig.savefig(root/'figures/longer_training.png',dpi=150)
    p=list(csv.DictReader((root/'results/singular_feature_paths/results.csv').open()))
    eps=sorted({float(r['epsilon']) for r in p});fig,axes=plt.subplots(1,2,figsize=(10,4))
    for field,label in [('minimum_norm_floor','Unrestricted span floor'),('ridge_error','Population ridge error')]:
        values=[statistics.median(float(r[field]) for r in p if float(r['epsilon'])==e) for e in eps]
        axes[0].plot(eps,values,'o-',label=label)
    for field,label in [('beta_norm','Unrestricted readout norm'),('ridge_beta_norm','Ridge readout norm')]:
        values=[statistics.median(float(r[field]) for r in p if float(r['epsilon'])==e) for e in eps]
        axes[1].plot(eps,values,'o-',label=label)
    for ax in axes:ax.set_xscale('symlog',linthresh=1e-5);ax.set_xlabel('Parameter interpolation fraction');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Population MSE');axes[1].set_ylabel('Readout coefficient norm');axes[1].set_yscale('log')
    fig.suptitle('Medians over 50 paths; unit-RMS feature columns; ridge λ=0.001')
    fig.tight_layout();fig.savefig(root/'figures/singular_feature_paths.png',dpi=150)
    (root/'results/readout_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
