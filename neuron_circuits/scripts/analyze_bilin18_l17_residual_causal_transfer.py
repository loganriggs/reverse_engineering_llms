import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition'
j=json.loads((ART/'bilin18_l17_residual_causal_transfer.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4));names=j['config']['models'];labels=['Affine / energy','Affine / sensitivity','Quadratic / energy','Quadratic / sensitivity']
for ax,metric,title in zip(axes,['residual_change_fidelity','relative_kl_improvement'],['Missing-response vector fidelity','Conditional KL improvement over compiler']):
 for offset,scale in [(-.18,.1),(.18,1.)]:
  vals=[100*j['results'][f'actual_s{scale}_pooled'][n][metric] for n in names];ax.bar([i+offset for i in range(4)],vals,width=.34,label=f'Edit scale {scale:g}')
 ax.set_xticks(range(4),labels,rotation=25,ha='right');ax.axhline(0,color='black',linewidth=.7);ax.set_title(title);ax.set_ylabel('Percent');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
fig.suptitle('Frozen natural-state predictors: actual L17 source interventions');fig.tight_layout()
for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_L17_RESIDUAL_CAUSAL_TRANSFER.{ext}',dpi=180)
