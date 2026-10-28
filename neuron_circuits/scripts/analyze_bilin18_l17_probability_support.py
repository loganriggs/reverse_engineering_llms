import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition'
j=json.loads((ART/'bilin18_l17_probability_support.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4.3))
for rank,ax in zip([16,64],axes):
    for group,color in [('all','#313b4b'),('low','#21867a'),('medium','#c69223'),('high','#b84d62')]:
        xs=[64,256,1024];ys=[100*j['results'][group][f'local_s{s}_r{rank}']['recovery'] for s in xs]
        ax.plot(xs,ys,'o-',label=f"{group} (n={j['selection'][group]['sites']})",color=color)
    ax.set_xscale('log',base=2);ax.set_xticks([64,256,1024],['64','256','1024']);ax.set_ylim(0,100);ax.set_title(f'Oracle correction: rank {rank}');ax.set_xlabel('Tokens retained in Fisher construction');ax.set_ylabel('Exact KL error recovered (%)');ax.grid(alpha=.2)
axes[0].legend(fontsize=8);fig.suptitle('Probability support and compiled-prediction entropy');fig.tight_layout()
for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_L17_PROBABILITY_SUPPORT.{ext}',dpi=180)
