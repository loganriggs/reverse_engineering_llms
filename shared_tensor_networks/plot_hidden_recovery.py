"""All36fits: output fidelity versus invariant hidden-subspace recovery."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).parent;rows=json.loads((root/'results/bilinear_hidden_recovery/results.json').read_text())['rows']
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for kind,label,color in [('tied_raw','Tied / identity damping','#2563eb'),('tied_corrected','Tied / occurrence damping','#9333ea'),('untied_raw','Untied / identity damping','#ea580c')]:
    selected=[r for r in rows if r['student']==kind]
    error=[max(r['population_mse'],1e-16) for r in selected]
    axes[0].scatter(error,[max(r['worst_angles_degrees']) for r in selected],label=label,color=color,s=30,alpha=.8)
    axes[1].scatter(error,[r['patches']['both']['difference_from_original_mse'] for r in selected],color=color,s=30,alpha=.8)
for ax in axes:
    ax.set_xscale('log');ax.axvline(1e-12,color='gray',linestyle=':',linewidth=1)
    ax.set_xlabel('Population prediction MSE');ax.grid(alpha=.2)
axes[0].set_ylabel('Worst branch principal angle (degrees)');axes[0].set_ylim(0,95)
axes[0].legend(fontsize=8,loc='lower left')
axes[1].set_yscale('log');axes[1].set_ylabel('Teacher-space replacement: output MSE')
fig.suptitle('Accurate function fits need not recover teacher hidden subspaces (36 fits)')
fig.savefig(root/'figures/bilinear_hidden_recovery.png',dpi=160)
