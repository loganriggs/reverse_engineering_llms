import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';fig,axes=plt.subplots(1,2,figsize=(11,4))
for i,(variant,label) in enumerate([('','Train even'),('_reverse','Train odd')]):
 j=json.loads((ART/f'bilin18_l17_response_supervision{variant}.json').read_text())
 for offset,group,color in [(-.18,'supervised','#218c80'),(.18,'heldout','#b44b61')]:
  axes[0].bar(i+offset,100*j['groups'][f'actual_s1.0_{group}']['quadratic1152_beta1']['residual_fidelity'],width=.34,color=color,label=group if i==0 else None)
axes[0].set_xticks([0,1],['Train even (5 heads)','Train odd (4 heads)']);axes[0].set_title('Reversing the head split');axes[0].legend()
for i,(variant,label) in enumerate([('','Actual'),('_angle','Same-head angle'),('_isotropic','Ambient angle')]):
 j=json.loads((ART/f'bilin18_l17_response_supervision{variant}.json').read_text())
 for offset,group,color in [(-.18,'supervised','#218c80'),(.18,'heldout','#b44b61')]:
  axes[1].bar(i+offset,100*j['groups'][f'actual_s1.0_{group}']['quadratic1152_beta1']['residual_fidelity'],width=.34,color=color,label=('even-head tests' if group=='supervised' else 'odd-head tests') if i==0 else None)
axes[1].set_xticks([0,1,2],['Actual source edits','Same-head random','Ambient random'],rotation=12);axes[1].set_title('Varying training directions (even norm donors)');axes[1].legend()
for ax in axes:ax.axhline(0,color='black',linewidth=.7);ax.set_ylabel('Actual-source response fidelity (%)');ax.set_ylim(-10,45);ax.grid(axis='y',alpha=.2)
fig.suptitle('Fixed quadratic features, primary response weight 1');fig.tight_layout()
for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_L17_RESPONSE_COVERAGE.{ext}',dpi=180)
