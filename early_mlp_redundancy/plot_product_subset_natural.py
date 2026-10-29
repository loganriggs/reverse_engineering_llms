"""Audited natural behavior versus local native-write error; no task outcomes used."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;V=P/'validation'
d=json.loads((V/'product_subset_natural_comparison.json').read_text());a=json.loads((V/'product_subset_natural_audit.json').read_text());assert a['status']=='passed';assert d['audit_sha256']==hashlib.sha256((V/'product_subset_natural_audit.json').read_bytes()).hexdigest()
colors={'energy':'#be5d26','random9001':'#2774a6','random9002':'#8453a0'}
fig,axs=plt.subplots(1,2,figsize=(11,4.9))
for ax,native in zip(axs,[True,False]):
 for r in d['comparisons']:
  if (r['space']=='native')!=native:continue
  raw,sc=r['raw'],r['scaled'];x=raw['local_relative_error'];y=raw['mean_KL'];xx=sc['local_relative_error'];yy=sc['mean_KL'];col=colors[r['method']]
  ax.scatter([x],[y],facecolors='white',edgecolors=col,s=32,alpha=.8);ax.scatter([xx],[yy],color=col,s=23,alpha=.8);ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=col,alpha=.6,lw=1))
 refs=[s for s in d['summaries'] if (s['space']=='native')==native and s['method']=='full']
 ax.scatter([s['local_relative_error'] for s in refs],[s['mean_KL'] for s in refs],marker='x',color='black',s=30,label='Full-product reference')
 ax.set_title('Native output space' if native else 'All 16 learned output spaces');ax.set_xlabel('Mean relative squared native-write error');ax.set_ylabel('Natural-text output KL');ax.grid(alpha=.18)
from matplotlib.lines import Line2D
handles=[Line2D([0],[0],color=c,label=m) for m,c in colors.items()]+[Line2D([0],[0],marker='x',color='black',lw=0,label='Full products')]
fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False)
fig.suptitle('Half of native products retained: raw (open) → frozen scalar compensation (filled)\nOutput KL and local error improve together here; causal fidelity remains untested',fontsize=12)
fig.tight_layout(rect=(0,.08,1,.91));fig.savefig(P/'figures/product_subset_natural.png',dpi=160);fig.savefig(P/'figures/product_subset_natural.svg');plt.close(fig)
