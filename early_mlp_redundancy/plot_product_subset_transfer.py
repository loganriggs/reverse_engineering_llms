from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;V=P/'validation'
train=json.loads((V/'product_subset_results.json').read_text())['rows'];test=json.loads((V/'product_subset_transfer_results.json').read_text())['pooled'];sizes=[128,512,2048,2304]
fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
for ax,metric,title in zip(axes,['raw_relative_squared','compensated_relative_squared'],['Raw deletion','Frozen training scalar compensation']):
 for data,style,split in [(train,'--','Training'),(test,'-','Held-out')]:
  native=[next(r[metric] for r in data if r['space']=='native' and r['method']=='energy' and r['size']==k) for k in sizes]
  groups=[[r[metric] for r in data if r['space']!='native' and r['method']=='energy' and r['size']==k] for k in sizes]
  lo=[min(g) for g in groups];hi=[max(g) for g in groups];mean=[np.mean(g) for g in groups]
  ax.plot(sizes,native,style,color='#245b94',marker='o',label=f'Native · {split}')
  ax.plot(sizes,mean,style,color='#b7542f',marker='s',label=f'Projected · {split}')
  ax.fill_between(sizes,lo,hi,color='#b7542f',alpha=.12)
 ax.set_title(title);ax.set_xlabel('Native products retained (of 4,608)');ax.set_xticks(sizes);ax.tick_params(axis='x',labelrotation=25);ax.grid(alpha=.2);ax.set_ylim(0,1)
axes[0].set_ylabel('Relative squared local error');axes[1].legend(fontsize=8)
fig.suptitle('Training-selected energy subsets transfer in local output geometry\nProjected lines: mean and range across all 16 decoder spaces; no causal claim',fontsize=12)
fig.tight_layout();fig.savefig(P/'figures/product_subset_transfer.png',dpi=160);fig.savefig(P/'figures/product_subset_transfer.svg');plt.close(fig)
