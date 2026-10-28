import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';j=json.loads((ART/'bilin18_l17_response_supervision.json').read_text());fig,axes=plt.subplots(1,2,figsize=(9,4))
for ax,payload in zip(axes,['actual','angle']):
 for group,color in [('supervised','#1f8b80'),('heldout','#b84b61')]:
  ax.plot(range(3),[100*j['groups'][f'{payload}_s1.0_{group}'][f'quadratic1152_beta{b}']['residual_fidelity'] for b in [0,1,10]],'o-',label=group+' heads',color=color)
 ax.set_xticks(range(3),['0','1','10']);ax.set_xlabel('Response training weight');ax.set_ylabel('Missing-response fidelity (%)');ax.axhline(0,color='black',linewidth=.7);ax.set_title('Actual head writes' if payload=='actual' else 'Norm/angle-matched directions');ax.legend();ax.grid(alpha=.2);ax.set_ylim(-15,50)
fig.suptitle('Fixed quadratic features: response supervision transfers unevenly');fig.tight_layout()
for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_L17_RESPONSE_SUPERVISION.{ext}',dpi=180)
