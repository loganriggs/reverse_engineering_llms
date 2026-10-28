#!/usr/bin/env python3
"""Display fixed payload contrasts, with pooled and equal-site weighting separated."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition'
def main():
    a=json.loads((ART/'bilin18_l8_execution_alignment.json').read_text());b=json.loads((ART/'bilin18_l8_angle_matched_alignment.json').read_text())
    fig,axs=plt.subplots(1,3,figsize=(15,4.6));colors=['#28649c','#b87529','#56836a','#8b5999'];families=['actual','shuffled','gaussian','angle_gaussian'];labels=['Actual','Other row','Gaussian','Angle matched']
    for ax,metric,title in [(axs[0],'pooled','Response-energy weighting'),(axs[1],'site_mean','Every site weighted equally')]:
        xs=np.arange(4);w=.18
        for j,f in enumerate(families):
            vals=[b['results'][f'{n}_{f}_full']['16'][metric] for n in ['native','right0','down0','direction0']]
            ax.bar(xs+(j-1.5)*w,vals,w,color=colors[j],label=labels[j])
        ax.set_xticks(xs,['Native','Input pair\nshuffle 0','Output\nshuffle 0','Direction\nshuffle 0']);ax.set_ylim(0,.65);ax.set_title(title);ax.set_ylabel('Rank-16 response coverage');ax.grid(axis='y',alpha=.2)
    ax=axs[2]
    for j,f in enumerate(families):
        vals=[b['head_results'][f'native_h{h}_{f}_full']['16'] for h in range(9)]
        ax.plot(range(9),vals,'o-',label=labels[j],color=colors[j],markersize=4)
    ax.set_xlabel('Source attention head');ax.set_ylabel('Rank-16 response coverage');ax.set_title('Native function, each source separately');ax.set_ylim(0,.85);ax.grid(alpha=.2)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.025));fig.suptitle('Does real source–context alignment survive matching RMS geometry?')
    fig.text(.5,.005,'Fresh 32-row follow-up; all 9 heads and 1,152 output coordinates. Full removal. Shuffles target their own functions.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.11,1,.95])
    for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_L8_EXECUTION_ALIGNMENT.{ext}',dpi=170)
    plt.close(fig)
if __name__=='__main__':main()
