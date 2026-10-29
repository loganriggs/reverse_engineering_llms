"""Plot fixed-feature controls against neural fits without choosing winners."""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).parent;fig,axes=plt.subplots(2,2,figsize=(11,7));summary=[]
    for col,(network,baseline) in enumerate([('e1_matched','e1_regression'),('e1_d8','e1_d8_regression')]):
        ns=list(csv.DictReader((root/'results'/network/'results.csv').open()))
        bs=list(csv.DictReader((root/'results'/baseline/'results.csv').open()))
        for row,teacher in enumerate(['tied','untied']):
            sizes=sorted({int(r['n']) for r in ns});ax=axes[row,col]
            for label,data,field,value in [('trained tied',ns,'student','tied'),('trained untied (same width)',ns,'student','untied_width'),
                                          ('quartic minimum norm',bs,'features','quartic_raw'),('oracle hidden minimum norm',bs,'features','oracle_hidden')]:
                medians=[]
                for n in sizes:
                    values=[float(r['exact_gaussian_mse']) for r in data if r['teacher']==teacher and int(r['n'])==n and r[field]==value and r.get('solver','minnorm')=='minnorm']
                    median=statistics.median(values);medians.append(median)
                    summary.append(dict(network=network,teacher=teacher,n=n,method=label,median=median,values=values))
                ax.plot(sizes,medians,'o-',label=label,markersize=4)
            ax.axhline(1,color='gray',ls=':',label='zero predictor')
            ax.set_xscale('log',base=2);ax.set_ylim(bottom=0);ax.grid(alpha=.2)
            ax.set_title(f'd={4 if col==0 else 8}, {teacher} teacher')
            ax.set_xlabel('Training examples');ax.set_ylabel('Exact Gaussian MSE')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Five-seed medians: function fitting versus learning the representation')
    fig.tight_layout();fig.savefig(root/'figures/regression_controls.png',dpi=150)
    (root/'results/regression_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
