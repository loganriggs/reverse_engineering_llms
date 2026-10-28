import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';j=json.loads((ART/'bilin18_l17_residual_tensor_partners.json').read_text());names=['core16','random0','random1','random2','within32','tensor'];fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,payload,title in zip(axes,['actual','angle'],['Actual source changes','Norm/angle controls']):
 vals=[100*j['results'][payload+'_s1.0'][n+'_selected']['fidelity'] for n in names];ax.bar(range(len(names)),vals,color=['#9aa6b2','#8194a5','#8194a5','#8194a5','#526e88','#208d80']);ax.set_xticks(range(6),['Core16','Random 1','Random 2','Random 3','Within32','Tensor'],rotation=25,ha='right');ax.set_title(title);ax.set_ylim(-5,55);ax.axhline(0,color='black',linewidth=.7);ax.set_ylabel('Missing-response fidelity (%)');ax.grid(axis='y',alpha=.2)
fig.suptitle('Mixed interaction partners for the remaining MLP error');fig.tight_layout()
for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_L17_RESIDUAL_TENSOR_PARTNERS.{ext}',dpi=180)
