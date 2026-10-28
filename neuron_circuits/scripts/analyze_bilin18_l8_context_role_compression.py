import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_context_role_compression'
a=json.loads(STEM.with_suffix('.json').read_text());rows=np.load(STEM.with_name(STEM.name+'_rows.npz'));ix=np.random.default_rng(20261103).integers(64,size=(2000,64));contrasts={}
for rule in ['context_only','cross']:
 diff=rows[f'pca64_{rule}_s1_error']-rows[f'operator64_{rule}_s1_error'];energy=rows['1_energy'];contrasts[rule]={'operator64_minus_pca64':float(diff.sum()/energy.sum()),'paired_row_bootstrap95':np.quantile(diff[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
STEM.with_name(STEM.name+'_contrasts.json').write_text(json.dumps(contrasts,indent=2))
fig,axs=plt.subplots(1,2,figsize=(10.5,4.3));ranks=[16,64,128,256,512]
for family,label,color in [('pca','PCA context code','#3267a8'),('operator','Operator-weighted code','#b64b36'),('random','Random code','#888888')]:
 axs[0].plot(ranks,[a['responses'][f'{family}{r}_s1']['cross'] for r in ranks],'o-',color=color,label=label)
 axs[1].plot(ranks,[a['natural_r2'][f'{family}{r}']['cross'] for r in ranks],'o-',color=color,label=label)
axs[0].plot(ranks,[a['responses'][f'operator{r}_s1']['context_only'] for r in ranks],'--',color='#b64b36',label='Operator code, context-only rule')
for ax in axs:
 ax.set_xscale('log',base=2);ax.set_xticks(ranks,[str(r) for r in ranks]);ax.set_ylim(0,1);ax.set_xlabel('Context-code rank');ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
axs[0].set_title('High response fidelity needs hundreds of reads');axs[0].set_ylabel('Full-head-removal response fidelity')
axs[1].set_title('Integrable cross-term state decoder');axs[1].set_ylabel('Natural full-write R²')
fig.text(.01,.01,'All native weights, full change vectors and exact normalization remain. Rank counts the context role only, not total model complexity.',fontsize=9)
fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(STEM.with_name(STEM.name.upper()+'.png'),dpi=180);fig.savefig(STEM.with_name(STEM.name.upper()+'.svg'))
