"""All fixed native canonical-margin cubes, with no selected examples."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_pca_encoder import V,sha


def main():
    ap=V/'repetition_cube_analysis.json';a=json.loads(ap.read_text());audit=json.loads((V/'repetition_cube_audit.json').read_text());assert audit['status']=='passed' and audit['analysis_sha256']==sha(ap);order=[0,4,2,6,1,5,3,7];data=np.array([x['canonical_margins'] for x in a['behavior']])[:,order];limit=np.max(np.abs(data));fig,ax=plt.subplots(figsize=(12,6));im=ax.imshow(data,cmap='coolwarm',vmin=-limit,vmax=limit,aspect='auto');ax.set_yticks(range(8),[f"{x['key']}: {x['values'][0]} / {x['values'][1]} ({x['format']})" for x in a['behavior']],fontsize=9);ax.set_xticks(range(8),['000','001','010','011','100','101','110','111']);ax.set_xlabel('Values at occurrences 1, 2, 3: 0 = first listed value; 1 = second listed value');ax.set_title('Native preferences on all fixed repetition cubes\nPositive margin favors the second value; mixed sequences have no imposed correct label')
    for i in range(8):
        for j in range(8):ax.text(j,i,f'{data[i,j]:.1f}',ha='center',va='center',fontsize=10,color='white' if abs(data[i,j])>limit*.6 else 'black')
    cb=fig.colorbar(im,ax=ax,pad=.025);cb.set_label('Canonical next-token logit difference (value1 − value0)');fig.tight_layout();fig.savefig(V.parent/'figures/repetition_cube.png',dpi=160);fig.savefig(V.parent/'figures/repetition_cube.svg');print('PLOTTED')
if __name__=='__main__':main()
