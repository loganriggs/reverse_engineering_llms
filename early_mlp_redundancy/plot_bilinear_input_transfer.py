"""M88 retention/causal-fidelity tradeoff for every input-model pair."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parent;V=P/'validation'
n=json.loads((V/'bilinear_input_natural.json').read_text());t=json.loads((V/'bilinear_input_factorial_comparison.json').read_text());nl={(x['kind'],x['seed'],x['arm']):x['mean_KL'] for x in n['summaries']};rows={(x['kind'],x['seed'],x['base_arm']):x for x in t['summaries'] if x['scope']=='contrast_qualified' and x['variant']=='input'}
colors=dict(dense='#2864a0',scalar='#df7c21',block='#25865a',shared_fixed='#9956ac');fig,axs=plt.subplots(1,2,figsize=(11,5.5),sharex=True)
for ax,site in zip(axs,['source','final']):
 for kind,color in colors.items():
  for seed,marker in [(71,'o'),(72,'s')]:
   xx=[nl[kind,seed,arm+'__input'] for arm in ['intact_KL','joint_three_corner']];yy=[rows[kind,seed,arm][site+'_reconstructed_reconstructed_family'] for arm in ['intact_KL','joint_three_corner']];ax.annotate('',xy=(xx[1],yy[1]),xytext=(xx[0],yy[0]),arrowprops=dict(arrowstyle='->',color=color,lw=1.4));ax.scatter(xx[0],yy[0],marker=marker,s=45,facecolors='white',edgecolors=color,zorder=3);ax.scatter(xx[1],yy[1],marker=marker,s=45,color=color,zorder=4)
 ax.axhline(1,color='#888',ls='--',lw=1);ax.text(.54,1.06,'Zero-response baseline',fontsize=9,color='#666');ax.set_yscale('log');ax.set_ylim(.85,60 if site=='source' else 22);ax.set_title(site.capitalize()+'-position interventions');ax.set_xlabel('Natural-text output KL (lower is better)');ax.grid(True,which='both',alpha=.15)
axs[0].set_ylabel('Relative causal-response error (lower is better)');fig.suptitle('Both background and change reconstructed\nArrows: KL-only → joint causal supervision; five contrast-qualified task families',fontsize=12)
handles=[Line2D([],[],color=c,label=k.replace('_',' ')) for k,c in colors.items()]+[Line2D([],[],color='gray',marker='o',linestyle='',label='seed 71'),Line2D([],[],color='gray',marker='s',linestyle='',label='seed 72')];fig.legend(handles=handles,loc='lower center',ncol=6,fontsize=9,frameon=False);fig.tight_layout(rect=[0,.08,1,.9]);dest=P/'figures';dest.mkdir(exist_ok=True);fig.savefig(dest/'bilinear_input_transfer.png',dpi=170);fig.savefig(dest/'bilinear_input_transfer.svg')
