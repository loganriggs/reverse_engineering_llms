"""M76 every paired encoder comparison, no task or seed selection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent

def main():
    result=json.loads((P/'validation/whole_task_factorial_comparison.json').read_text());rows=result['summaries'];lookup={(r['kind'],r['seed'],r['base_arm'],r['variant'],r['scope']):r for r in rows}
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    styles={'scalar':('o','#0072b2'),'block':('^','#009e73'),'shared_fixed':('s','#cc79a7')}
    for ax,site,corner in zip(axes.flat,['source','final','source','final'],['native_reconstructed','native_reconstructed','reconstructed_reconstructed','reconstructed_reconstructed']):
        metric=site+'_'+corner+'_family'
        for r in rows:
            if r['scope']!='all' or r['variant']!='original':continue
            d=lookup[r['kind'],r['seed'],r['base_arm'],'full_dictionary_LS','all'];marker,color=styles[r['kind']];ax.plot([0,1],[r[metric],d[metric]],color=color,alpha=.6,marker=marker,lw=1)
        for r in rows:
            if r['scope']=='all' and r['kind']=='pca':ax.scatter(2 if r['arm']=='rank32' else 3,r[metric],marker='*',c='#d55e00',s=100)
        ax.axhline(1,ls='--',color='gray');ax.set(xticks=[0,1,2,3],xticklabels=['Sparse','Dense LS','PCA32','PCA128'],ylabel='Equal-family response error',title=site.capitalize()+': '+('native background' if corner=='native_reconstructed' else 'reconstructed background'),yscale='log');ax.grid(alpha=.2)
    handles=[plt.Line2D([],[],color=c,marker=m,label=k.replace('shared_fixed','shared dictionary')) for k,(m,c) in styles.items()]
    fig.legend(handles=handles,loc='outside lower center',ncol=3);fig.suptitle('All seven training-excluded task families; lower is better\nPaired lines retain every dictionary/seed/objective; dashed line = zero-response baseline')
    fig.savefig(P/'figures/whole_task_factorial.png',dpi=180);fig.savefig(P/'figures/whole_task_factorial.svg')
if __name__=='__main__':main()
