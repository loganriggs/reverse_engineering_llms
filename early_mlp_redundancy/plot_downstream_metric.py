"""M74 fixed descriptive plots of all19 representations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
P=Path(__file__).resolve().parent

def main():
    data=json.loads((P/'validation/downstream_metric_comparison.json').read_text())['summaries']
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    styles={'pca':('*','#d55e00',140),'scalar':('o','#0072b2',35),'block':('^','#009e73',40),'shared_fixed':('s','#cc79a7',35)}
    for row in data:
        marker,color,size=styles[row['kind']]
        y=row['finite'][-1]['KL']
        axes[0].scatter(row['relative_residual_energy'],y,marker=marker,c=color,s=size)
        axes[1].scatter(row['Fisher_energy']/2,y,marker=marker,c=color,s=size)
        axes[2].plot([x['alpha'] for x in row['finite']],[x['actual_over_Taylor'] for x in row['finite']],marker=marker,c=color,alpha=1 if row['kind']=='pca' else .55,lw=2 if row['kind']=='pca' else 1)
    axes[0].set(xlabel='Mean relative squared write error',ylabel='Full replacement output KL',title='Euclidean geometry')
    axes[1].set(xlabel='Local Taylor prediction at full strength',ylabel='Full replacement output KL',title='Downstream sensitivity')
    axes[2].axhline(1,c='gray',ls='--');axes[2].set(xlabel='Replacement strength',ylabel='Actual KL / Taylor KL',title='Finite replacement effects',xticks=[.25,.5,1])
    for ax in axes:ax.grid(alpha=.2)
    axes[1].xaxis.set_major_locator(MaxNLocator(5))
    handles=[plt.Line2D([],[],color=c,marker=m,linestyle='',label=('PCA128' if k=='pca' else ('shared dictionary' if k=='shared_fixed' else k)+' dense128')) for k,(m,c,s) in styles.items()]
    fig.legend(handles=handles,loc='outside lower center',ncol=4)
    fig.suptitle('MLP1: fixed 128-dimensional representations on 32 natural documents')
    dest=P/'figures';dest.mkdir(exist_ok=True)
    fig.savefig(dest/'downstream_residual_metric.png',dpi=180);fig.savefig(dest/'downstream_residual_metric.svg')
if __name__=='__main__':main()
