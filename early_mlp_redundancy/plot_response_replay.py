"""M21 scientific diagnostic figure, one post-selected sentinel."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;d=json.loads((P/'validation/response_trajectory_replay.json').read_text());rows=d['records'];x=np.arange(121);loss=np.array([rows[0]['sentinel_before']]+[r['sentinel_after'] for r in rows]);out=P/'figures';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True,layout='constrained')
axes[0].semilogy(x,loss,color='#2459a6',label='Observed target response error');changed=[i+1 for i,r in enumerate(rows) if r['changed_tokens']];axes[0].scatter(changed,loss[changed],marker='x',color='#b45a00',s=32,label='Support changed during update',zorder=3);axes[0].axvline(80,color='gray',lw=1,ls=':');axes[0].set_ylabel('Response error (log scale)');axes[0].legend(loc='upper left',fontsize=9);axes[0].set_title('Exact block71 replay: one training target deteriorates after update 80')
for key,label,color,style in [('total_change','Observed net change','#2459a6','-'),('smooth_change','Change with previous support fixed','#30935a','-'),('gate_change','Additional support-change effect','#b45a00','-'),('first_order_prediction','Gradient · actual parameter update','#9a4ca0','--')]:axes[1].plot(x,np.r_[0,np.cumsum([r[key] for r in rows])],label=label,color=color,ls=style)
axes[1].axhline(0,color='gray',lw=.6);axes[1].set_ylabel('Cumulative change in response error');axes[1].set_xlabel('Completed updates');axes[1].legend(loc='upper left',fontsize=9)
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Post-result diagnosis; one sentinel, unchanged failed training run',fontsize=11)
fig.savefig(out/'response_replay.png',dpi=180);fig.savefig(out/'response_replay.svg');plt.close(fig)
print(out/'response_replay.png')
