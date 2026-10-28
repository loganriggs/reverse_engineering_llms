#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_context_response_geometry'
a=json.loads(STEM.with_suffix('.json').read_text());ranks=a['config']['ranks'];fig,axs=plt.subplots(1,2,figsize=(10.5,4.3))
for family,label,color in [('matched','Current state + source','#b64b36'),('wrong_context','Wrong context','#888888'),('wrong_source','Wrong source','#559176'),('global','Frozen global basis','#3267a8')]:
 for ax,metric in zip(axs,['coverage','prediction_fidelity']):
  ax.plot(ranks,[a['results']['1'][f'{family}{r}'][metric] for r in ranks],'o-',label=label,color=color)
axs[1].axhline(a['results']['1']['full_tangent']['prediction_fidelity'],c='#b64b36',ls=':',lw=1,label='Full native tangent')
for ax in axs:
 ax.set_xscale('log',base=2);ax.set_xticks(ranks,[str(r) for r in ranks]);ax.set_ylim(0,1.05);ax.set_xlabel('Output rank');ax.spines[['top','right']].set_visible(False);ax.legend(frameon=True,facecolor='white',edgecolor='white',framealpha=1,fontsize=8)
axs[0].set_title('Current-context spaces capture more response');axs[0].set_ylabel('Oracle coverage of exact finite response')
axs[1].set_title('Tangent coefficients retain much of that gain');axs[1].set_ylabel('Finite-response prediction fidelity')
fig.text(.01,.01,'Full head removals, all 9 heads, 128 sites. Charts use native weights/state and expensive SVD; no cheap decoder or semantic identity claimed.',fontsize=9)
fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(STEM.with_name(STEM.name.upper()+'.png'),dpi=180);fig.savefig(STEM.with_name(STEM.name.upper()+'.svg'))
