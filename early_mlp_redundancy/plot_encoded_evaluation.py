"""Fixed all-candidate summary of M70 causal and retention outcomes."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
r=json.loads((P/'validation/encoded_change_evaluation_comparison.json').read_text());rows=r['rows'];groups=[(k,s) for k in ['scalar','block','shared_fixed'] for s in [71,72]];arms=['intact_KL','KL_balanced_response','KL_encoded_balanced_response'];labels=['KL only','Supplied-change loss','Encoded-change loss'];colors=['#8a96a3','#d68830','#236ba1'];fig,axes=plt.subplots(1,3,figsize=(14,4.4));x=np.arange(6)
for ax,site in zip(axes[:2],['source','final']):
 for i,(arm,label,col) in enumerate(zip(arms,labels,colors)):
  vals=[next(t for t in rows if (t['kind'],t['seed'],t['arm'])==(k,s,arm))[f'holdout_{site}_native_reconstructed_family'] for k,s in groups];ax.bar(x+(i-1)*.24,vals,.23,label=label,color=col)
 ax.axhline(1,color='#333333',ls='--',lw=1);ax.set_title(f'{site.capitalize()} position: encoded-direction error');ax.set_ylabel('Equal-family response error (lower is better)');ax.set_ylim(0,3.5);ax.grid(axis='y',alpha=.15)
for i,(arm,col) in enumerate(zip(arms[1:],colors[1:])):
 vals=[]
 for k,s in groups:
  base=next(t for t in rows if (t['kind'],t['seed'],t['arm'])==(k,s,arms[0]))['natural_mean_KL'];value=next(t for t in rows if (t['kind'],t['seed'],t['arm'])==(k,s,arm))['natural_mean_KL'];vals.append(100*(value/base-1))
 axes[2].bar(x+(i-.5)*.3,vals,.29,color=col)
axes[2].axhline(0,color='#333333',lw=1);axes[2].set_title('Natural-text KL versus KL-only control');axes[2].set_ylabel('Relative change (%) — lower is better');axes[2].grid(axis='y',alpha=.15)
for ax in axes:
 ax.set_xticks(x,[f'{k.replace("shared_fixed","shared")}\n{s}' for k,s in groups]);ax.tick_params(axis='x',labelsize=9);ax.spines[['top','right']].set_visible(False)
handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,loc='upper center',ncol=3,frameon=False);fig.subplots_adjust(top=.81,bottom=.2,wspace=.32);fig.text(.02,.025,'14 familiar source-group evaluation pairs; all six architecture/seed pairs retained. Dashed line = zero response. Natural retention: 32 fixed documents.',fontsize=9);out=P/'figures';out.mkdir(exist_ok=True);fig.savefig(out/'encoded_change_evaluation.png',dpi=170);fig.savefig(out/'encoded_change_evaluation.svg');print(out/'encoded_change_evaluation.png')
