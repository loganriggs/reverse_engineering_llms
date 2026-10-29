"""Fresh-teacher means; show every teacher, never hide failed optimizer seeds."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent;s=json.loads((root/'results/finite_readout_confirmation/summary.json').read_text());rows=s['aggregates']
fig,axes=plt.subplots(1,2,figsize=(9,4.1),sharey=True)
for ax,budget in zip(axes,[384,1536]):
 for i,seed in enumerate(range(222001,222006)):
  vals=[next(r['transfer'] for r in rows if r['budget']==budget and r['initialization']=='random' and r['teacher_seed']==seed and r['heads']==h) for h in [1,3,6]]
  ax.plot([0,1,2],vals,'o-',color=f'C{i}',label=str(seed),alpha=.85)
 ax.set_xticks([0,1,2],['1','3','6']);ax.set_xlabel('Supervised source heads');ax.set_title(f'{budget:,} source scalar labels');ax.set_yscale('log');ax.grid(alpha=.2)
axes[0].set_ylabel('Held-out transfer MSE (log scale)')
fig.suptitle('Fresh-teacher confirmation: fixed label budget',y=.98)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,title='Teacher seed; each point averages three initializations')
fig.tight_layout(rect=(0,.18,1,.94));fig.savefig(root/'figures/finite_readout_confirmation.png',dpi=170);plt.close(fig)
