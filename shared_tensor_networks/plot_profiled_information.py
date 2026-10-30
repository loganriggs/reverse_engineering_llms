import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path('results/profiled_information');d=json.loads((root/'checks.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
r=d['near_degeneracy'][1:]
axes[0].loglog([x['delta'] for x in r],[x['smallest_singular'] for x in r],'o-',color='#276b9e')
axes[0].set(xlabel='Distance parameter from degenerate dataset',ylabel='Smallest profiled singular value',title='Full rank can be arbitrarily weak')
for seed in sorted({r['seed'] for r in d['linear_noise']}):
 rows=[r for r in d['linear_noise'] if r['seed']==seed]
 axes[1].loglog([r['n'] for r in rows],[r['predicted_squared_error'] for r in rows],'-',alpha=.6)
 axes[1].scatter([r['n'] for r in rows],[r['observed_squared_error'] for r in rows],marker='x',s=24,color='black')
axes[1].set(xlabel='Number of inputs (three tasks)',ylabel='Expected squared local-coordinate error',title='Linear noise risk: theory and simulation')
axes[1].plot([],[],color='gray',label='Predicted; five input/readout fixtures')
axes[1].plot([],[],'x',color='black',label='4,000 noise draws per point')
axes[1].legend(fontsize=8)
axes[1].set_xticks([16,64,256],labels=['16','64','256'])
axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
for ax in axes:ax.grid(True,which='both',alpha=.2)
fig.savefig(root/'conditioning.png',dpi=170)
plt.close(fig)
