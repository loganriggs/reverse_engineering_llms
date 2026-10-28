import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';j=json.loads((ART/'bilin18_l17_response_width_data.json').read_text());fig,axes=plt.subplots(1,2,figsize=(9,4))
for width,color in [(32,'#667d91'),(64,'#208d80')]:
 for ax,which in zip(axes,['train','test']):
  vals=[]
  for n in [256,768]:
   cell=f'w{width}_n{n}';vals.append(100*(j['audits'][cell]['fits']['0']['training_fidelity'] if which=='train' else j['results']['actual_s1.0'][cell+'_selected']['fidelity']))
  ax.plot([256,768],vals,'o-',label=f'Quadratic width {width}',color=color)
for ax,title in zip(axes,['Unregularized empirical training fit','Validation-selected test fidelity']):
 ax.set_xticks([256,768]);ax.set_ylim(0,100);ax.set_xlabel('Documents with response supervision');ax.set_ylabel('Missing-response fidelity (%)');ax.set_title(title);ax.legend();ax.grid(alpha=.2)
fig.suptitle('Fixed PCA geometry: more labels versus more quadratic products');fig.tight_layout()
for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_L17_RESPONSE_WIDTH_DATA.{ext}',dpi=180)
