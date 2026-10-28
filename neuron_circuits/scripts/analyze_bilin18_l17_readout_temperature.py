import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';j=json.loads((ART/'bilin18_l17_readout_temperature.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,scale in zip(axes,[.1,1.]):
    for tc in [.5,1.,2.]:
        vals=[100*j['results'][f'scale{scale}_eval{te}_chart{tc}_r16']['recovery'] for te in [.5,1.,2.]]
        ax.plot([.5,1.,2.],vals,'o-',label=f'Chart temperature {tc:g}')
    ax.set(xlabel='Evaluation temperature',ylabel='Exact KL error recovered (%)',title=f'Fixed internal error × {scale:g}',ylim=(0,100));ax.set_xscale('log',base=2);ax.set_xticks([.5,1.,2.],['0.5','1','2']);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Same internal computation, altered readout: rank16 oracle correction');fig.tight_layout()
for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_L17_READOUT_TEMPERATURE.{ext}',dpi=180)
