#!/usr/bin/env python3
"""Post-result paired contrasts and scope visualization for full-output audit."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_response_population'
a=json.loads(STEM.with_suffix('.json').read_text());rows=np.load(STEM.with_name(STEM.name+'_finite_rows.npz'));ix=np.random.default_rng(20261101).integers(32,size=(2000,32));contrast={}
for h in ['1','3']:
 contrast[h]={}
 for kind,e,en in [('logits','error','energy'),('ce','ce_error','ce_energy')]:
  diff=rows[f'h{h}_natural64_{e}']-rows[f'h{h}_response_all64_{e}'];energy=rows[f'h{h}_{en}']
  contrast[h][kind]={'gain':float(diff.sum()/energy.sum()),'paired_row_bootstrap95':np.quantile(diff[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
STEM.with_name(STEM.name+'_downstream_contrasts.json').write_text(json.dumps(contrast,indent=2))
fig,axs=plt.subplots(1,2,figsize=(11,4.3));ranks=[4,16,64,128]
for n,label,color in [('natural','Natural-write basis','#3267a8'),('response_all','All-head response basis','#b64b36'),('weight','Weight-output basis','#559176')]:
 vals=[a['population'][f'{n}{r}']['response_coverage'] for r in ranks]
 axs[0].plot(ranks,vals,'o-',label=label,color=color)
axs[0].scatter([4],[a['population']['selected4']['response_coverage']],marker='*',s=130,c='#8064a2',label='Four studied axes',zorder=5)
axs[0].set_xlabel('Output rank');axs[0].set_ylabel('Full MLP response energy captured');axs[0].set_ylim(0,.5);axs[0].set_title('Fixed low-rank spaces omit most response energy');axs[0].legend(frameon=False,fontsize=9)
heads=np.arange(9)
axs[1].bar(heads-.18,[a['head_statistics'][str(h)]['full_energy_share'] for h in heads],width=.36,label='Full output',color='#3267a8')
axs[1].bar(heads+.18,[a['head_statistics'][str(h)]['selected_energy_share'] for h in heads],width=.36,label='Within four studied axes',color='#b64b36')
axs[1].set_xticks(heads);axs[1].set_xlabel('Removed attention head');axs[1].set_ylabel('Share of summed response energy');axs[1].set_title('Head 3 is overrepresented by axis selection');axs[1].legend(frameon=False,fontsize=9)
for ax in axs:ax.spines[['top','right']].set_visible(False)
fig.text(.01,.01,'L8, 512 fresh sites. Output projections use exact responses; they are coverage diagnostics, not executable compressed programs.',fontsize=9)
fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(STEM.with_name('BILIN18_L8_RESPONSE_POPULATION.png'),dpi=180);fig.savefig(STEM.with_name('BILIN18_L8_RESPONSE_POPULATION.svg'))
