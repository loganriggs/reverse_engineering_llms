import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition'
diagnostics={}
for stem in ['bilin18_l8_channel_coordination','bilin18_l8_canonical_channel_coordination']:
 a=json.loads((ART/f'{stem}.json').read_text());stats={}
 for key in a['results']:
  if not key.endswith('_s1'):continue
  n=key[:-3]
  stats[n]={'macro_full_matched16':float(np.mean([a['head_results'][f'{n}_h{h}_s1']['matched16']['coverage'] for h in range(9)])),
            'macro_small_matched16':float(np.mean([a['head_results'][f'{n}_h{h}_s0.05']['matched16']['coverage'] for h in range(9)])),
            'macro_isotropic_spectrum16':float(np.mean([a['isotropic_spectral_capture'][n][str(h)]['16'] for h in range(9)]))}
 diagnostics[stem]=stats
(ART/'bilin18_l8_channel_coordination_macro_audit.json').write_text(json.dumps({'scope':'Post-result uniform-head averages. Isotropic spectrum and actual response coverage weight states differently; their difference alone is not a pure directional-alignment estimate.','results':diagnostics},indent=2))
a=json.loads((ART/'bilin18_l8_canonical_channel_coordination.json').read_text());fig,axs=plt.subplots(1,2,figsize=(11,4.4))
groups=[('Native',['native']),('Joint\npermutation',['joint']),('Input-pair\nshuffle',[f'right{r}' for r in range(3)]),('Output\nshuffle',[f'down{r}' for r in range(3)]),('Direction\nshuffle',[f'direction{r}' for r in range(3)])]
colors=['#b64b36','#a87266','#3267a8','#559176','#8064a2']
for i,((label,names),color) in enumerate(zip(groups,colors)):
 for ax,metric in zip(axs,['matched16','pca64_cross']):
  vals=[a['results'][n+'_s1'][metric]['coverage' if metric=='matched16' else 'prediction_fidelity'] for n in names]
  xs=[i] if len(vals)==1 else np.linspace(i-.12,i+.12,len(vals))
  ax.scatter(xs,vals,c=color);ax.hlines(np.mean(vals),i-.28,i+.28,color=color,lw=2)
for ax in axs:
 ax.set_xticks(range(5),[label for label,_ in groups],fontsize=9);ax.set_ylim(0,.55);ax.spines[['top','right']].set_visible(False)
axs[0].set_title('Native bindings concentrate actual responses');axs[0].set_ylabel('State-matched rank 16 response coverage')
axs[1].set_title('A small shared context code remains inadequate');axs[1].set_ylabel('PCA context 64 cross-rule prediction fidelity')
fig.text(.01,.01,'Scaling/sign/pair-order canonicalized. Each score targets its own local function; upstream states are trained. Dots are permutation controls.',fontsize=9)
fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(ART/'BILIN18_L8_CHANNEL_COORDINATION_CONTROLS.png',dpi=180);fig.savefig(ART/'BILIN18_L8_CHANNEL_COORDINATION_CONTROLS.svg')
