import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('results/sampled_hidden_information');records=json.loads((root/'checks.json').read_text())['records']
n=np.arange(2,16);naive=np.minimum(12,3*np.maximum(n-3,0));combined=np.minimum(naive,2*np.maximum(n-2,0))
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
ax.plot(n,naive,'--',color='#3973ac',label='Observation minus readout bound')
ax.plot(n,combined,color='#d27a16',linewidth=2,label='Including sampled basis-change ambiguity')
sample=sorted({r['n'] for r in records});observed=[next(r['observed_ranks'][0] for r in records if r['n']==k and r['tasks']==3) for k in sample]
ax.scatter(sample,observed,color='#222222',s=40,zorder=3,label='Measured rank (all 3 seeds agree)')
ax.set(xlabel='Number of common input samples',ylabel='Distinguishable hidden tangent directions',title='Sampled hidden information: d=4, h=2, three tasks',ylim=(-.5,13),xticks=[2,3,4,6,7,8,10,12,15])
ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=9)
fig.savefig(root/'rank_bounds.png',dpi=180);fig.savefig(root/'rank_bounds.svg');plt.close(fig)
