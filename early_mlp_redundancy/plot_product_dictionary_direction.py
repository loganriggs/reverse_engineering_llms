"""M97 source-response direction: common training normalization, all shared families."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_pca_encoder import V

def main():
 train=json.loads((V/'product_dictionary_direction_training_comparison_v2.json').read_text());held=json.loads((V/'product_dictionary_direction_source_comparison_v2.json').read_text());scales=json.loads((V/'balanced_response_training_protocol.json').read_text())['scales'];den={(s['family'],s['region']):s['denominator'] for s in scales};families=sorted({r['family'] for r in held['summaries'] if r['family']});fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
 for ax,seed in zip(axes,[71,72]):
  lookup=lambda data,f:next(r['raw_mean'] for r in data['branch_summaries'] if r['seed']==seed and r['family']==f and r['region']=='changed_input_positions' and r['corner']=='reconstructed_reconstructed')
  y=np.arange(len(families));tr=[lookup(train,f)['slope0']/den[f,'changed_input_positions'] for f in families];hs=[lookup(held,f)['slope0']/den[f,'changed_input_positions'] for f in families];hf=[lookup(held,f)['finite_change']/den[f,'changed_input_positions'] for f in families]
  ax.scatter(tr,y-.15,color='#2176ae',label='Training: initial slope',s=45);ax.scatter(hs,y,color='#d16b29',label='Held-out: initial slope',s=50);ax.scatter(hf,y+.15,color='#222222',marker='x',label='Held-out: finite change',s=45);ax.axvline(0,color='#888888',linewidth=1);ax.set_yticks(y,[f.replace('_',' ') for f in families]);ax.invert_yaxis();ax.set_title(f'Seed {seed}');ax.grid(axis='x',alpha=.2);ax.set_xlabel('Source response-error change / fixed training denominator')
 axes[0].legend(loc='lower left',fontsize=8,frameon=False);fig.suptitle('Local directions already predict the held-out source regressions',fontsize=14);fig.text(.5,.015,'Negative helps; positive harms. Both background and change replaced. Each family uses the same denominator across datasets.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(V.parent/'figures/product_dictionary_direction.png',dpi=170);fig.savefig(V.parent/'figures/product_dictionary_direction.svg');plt.close(fig)
if __name__=='__main__':main()
