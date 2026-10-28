import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition'
j=json.loads((ART/'bilin18_l17_residual_feature_objective.json').read_text())
fig,ax=plt.subplots(figsize=(8,4));families=['affine1152','affine624','quadratic1152']
for offset,obj,color in [(-.18,'energy','#65788c'),(.18,'sensitivity','#1e8f84')]:
 ax.bar([i+offset for i in range(3)],[100*j['results'][j['selected'][f+'_'+obj]]['kl_recovery'] for f in families],width=.34,label=obj,color=color)
ax.set_xticks(range(3),['Affine 1152','Affine 624 control','624 linear + 528 quadratic']);ax.set_ylabel('Native KL error recovered (%)');ax.set_ylim(0,100);ax.legend();ax.set_title('Frozen predictors on 128 new documents');ax.grid(axis='y',alpha=.2);fig.tight_layout()
for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_L17_RESIDUAL_FEATURE_OBJECTIVE.{ext}',dpi=180)
