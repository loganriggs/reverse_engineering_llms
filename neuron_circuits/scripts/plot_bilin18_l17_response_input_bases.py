import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';j=json.loads((ART/'bilin18_l17_response_input_bases.json').read_text());fig,ax=plt.subplots(figsize=(8,4));names=['natural','source','residual_jacobian']
for offset,payload,color in [(-.18,'actual','#218f81'),(.18,'angle','#6b7c94')]:
 vals=[100*j['results'][payload+'_s1.0'][n+'_selected']['fidelity'] for n in names];ax.bar([i+offset for i in range(3)],vals,width=.34,label='Actual sources' if payload=='actual' else 'Angle controls',color=color)
ax.set_xticks(range(3),['Natural-state PCA','Source-displacement PCA','Residual-Jacobian basis']);ax.set_ylim(-20,60);ax.axhline(0,color='black',linewidth=.7);ax.set_ylabel('Missing-response fidelity (%)');ax.set_title('Fixed width32, 528 products, identical linear624 features');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout()
for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_L17_RESPONSE_INPUT_BASES.{ext}',dpi=180)
