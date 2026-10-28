#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ART=Path(__file__).resolve().parents[1]/'artifacts/bilin18_tensor_decomposition'
def main():
    p=json.loads((ART/'bilin18_l8_source_self_inclusion.json').read_text());t=json.loads((ART/'bilin18_l8_source_terms.json').read_text())
    fig,axs=plt.subplots(1,3,figsize=(15,4.8));colors=['#28649c','#b87529','#56836a','#8b5999'];fams=['actual','shuffled','gaussian','angle'];labels=['Actual','Other row','Gaussian','Angle matched'];conditions=[('intact','tangent'),('removed','tangent'),('removed','add')]
    for ax,metric,title in [(axs[0],'pooled','Response-energy weighting'),(axs[1],'site_mean','Every site weighted equally')]:
        for i,f in enumerate(fams):
            ax.bar(np.arange(3)+(i-1.5)*.19,[p['results'][f'native_{b}_{f}_{g}']['16'][metric] for b,g in conditions],.19,color=colors[i],label=labels[i])
        ax.set_xticks(range(3),['Intact\ntangent','Head removed\ntangent','Head removed\nfull addition']);ax.set_ylabel('Rank-16 response coverage');ax.set_title(title);ax.set_ylim(0,.75);ax.grid(axis='y',alpha=.2)
    ax=axs[2];terms=['cross','self','normalization'];names=['Context × head','Head × head','Normalization'];values=[t['pooled']['native'][k]['signed_projection'] for k in terms]
    ax.bar(range(3),values,color=['#28649c','#b87529','#8b5999']);ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3),names,rotation=15);ax.set_ylabel('Signed projection onto actual response');ax.set_title('Exact additive terms (non-unique credits)');ax.grid(axis='y',alpha=.2)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.34,.03),ncol=4)
    fig.suptitle('Does alignment survive removing the source from the chart baseline?')
    fig.text(.5,.007,'32 rows from ONE document × 4 positions; all 9 heads / 1,152 outputs. Intervened baselines; nonorthogonal terms.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.14,1,.94])
    for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_L8_SOURCE_SELF_INCLUSION.{ext}',dpi=170)
if __name__=='__main__':main()
