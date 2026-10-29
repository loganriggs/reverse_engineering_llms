"""Plot all frozen role comparisons; null ranges are not confidence intervals."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_pca_encoder import V,sha


def main():
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    categories=[('intact_KL','trained'),('KL_encoded_balanced_response','trained'),('intact_KL','shuffled'),('KL_encoded_balanced_response','shuffled')]
    for col,cohort in enumerate(['train','source','whole']):
        cp=V/f'block_change_roles_{cohort}_comparison.json';a=json.loads((V/f'block_change_roles_{cohort}_comparison_audit.json').read_text());assert a['status']=='passed' and a['comparison_sha256']==sha(cp);c=json.loads(cp.read_text())
        for row,region in enumerate(['changed_input_positions','final_position']):
            ax=axes[row,col]
            for x,(arm,grouping) in enumerate(categories):
                r=next(x for x in c['comparisons'] if x['arm']==arm and x['grouping']==grouping and x['region']==region and x['scope']=='all');n=r['random_summary']['relative_error'];ax.plot([x+.12,x+.12],[n['q05'],n['q95']],color='#8c929a',lw=2);ax.scatter(x+.12,n['mean'],color='#8c929a',s=25,label='Random assignment mean; 5–95% range' if x==0 else None);ax.scatter(x-.12,r['geometry']['relative_error'],color='#1769aa',s=40,label='Fixed geometry assignment' if x==0 else None)
            ax.axhline(1,color='#a43b31',ls='--',lw=1);ax.set_yscale('log');ax.grid(axis='y',alpha=.2);ax.set_xticks(range(4),['KL\nblocks','Causal\nblocks','KL\nshuffled','Causal\nshuffled']);ax.set_title({'train':'Training families (11 pairs)','source':'Source-group holdout (14 pairs)','whole':'Whole-task holdout (56 pairs)'}[cohort]+'\n'+('Source intervention' if row==0 else 'Final-position intervention'));ax.set_ylabel('Relative squared effect error' if col==0 else '')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.03));fig.suptitle('Do geometrically matched blocks have reproducible task-change roles?',fontsize=15);fig.text(.5,.005,'Lower is better. Dashed line: predict zero effect. Random ranges describe fixed permutations, not statistical confidence.',ha='center',fontsize=10);fig.tight_layout(rect=(0,.10,1,.94));dest=V.parent/'figures';dest.mkdir(exist_ok=True);fig.savefig(dest/'block_change_roles.png',dpi=160);fig.savefig(dest/'block_change_roles.svg');print('PLOTTED')
if __name__=='__main__':main()
