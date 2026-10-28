import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';j=json.loads((ART/'bilin18_l17_response_capacity.json').read_text());fit=j['fit_diagnostics'];fig,ax=plt.subplots(figsize=(8,4.2))
for offset,family,color in [(-.18,'affine1152','#61788b'),(.18,'quadratic1152','#218d81')]:
 key=fit['selected'][family+'_response'];alpha=key.split('_alpha')[-1];vals=[fit['diagnostics'][family]['fits']['0']['training_fidelity'],fit['validation'][family+'_alpha0']['response_fidelity'],fit['diagnostics'][family]['fits'][alpha]['training_fidelity'],fit['validation'][key]['response_fidelity'],j['results']['actual_s1.0_pooled'][family+'_response']['residual_change_fidelity']]
 ax.bar([i+offset for i in range(5)],[100*v for v in vals],width=.34,label=family,color=color)
ax.set_xticks(range(5),['Unregularized\ntrain','Unregularized\nvalidation','Selected ridge\ntrain','Selected ridge\nvalidation','Selected ridge\ntest']);ax.set_ylim(0,100);ax.set_ylabel('Missing-response fidelity (%)');ax.set_title('All-head response-only fitting: capacity and generalization');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout()
for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_L17_RESPONSE_CAPACITY.{ext}',dpi=180)
