#!/usr/bin/env python3
"""Document replication comparison and response-energy concentration audit."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ART=Path(__file__).resolve().parents[1]/'artifacts/bilin18_tensor_decomposition'
def main():
    single=json.loads((ART/'bilin18_l8_source_self_inclusion.json').read_text());multi=json.loads((ART/'bilin18_l8_document_replication.json').read_text());rows=np.load(ART/'bilin18_l8_document_replication_rows.npz');audit={}
    for b,g in [('intact','tangent'),('intact','remove'),('removed','tangent'),('removed','add')]:
        prefix=f'native_{b}_actual_{g}';angle=f'native_{b}_angle_{g}';e=rows[prefix+'_energy'];cap=rows[prefix+'_r16_captured'];ae=rows[angle+'_energy'];ac=rows[angle+'_r16_captured'];w=e/e.sum()
        loo=(cap.sum()-cap)/(e.sum()-e)-(ac.sum()-ac)/(ae.sum()-ae)
        audit[f'{b}_{g}']={'positive_document_gaps':int(np.sum(cap/e-ac/ae>0)),'document_gap_median':float(np.median(cap/e-ac/ae)),'document_gap_q10_q90':np.quantile(cap/e-ac/ae,[.1,.9]).tolist(),'largest_document_energy_share':float(w.max()),'top4_document_energy_share':float(np.sort(w)[-4:].sum()),'energy_weight_effective_documents':float(1/(w*w).sum()),'actual_minus_angle_leave_one_document_out_min_max':[float(loo.min()),float(loo.max())]}
    (ART/'bilin18_l8_document_replication_energy_audit.json').write_text(json.dumps(audit,indent=2))
    fig,axs=plt.subplots(1,3,figsize=(15,4.8));conditions=[('intact','tangent'),('removed','tangent'),('removed','add')];labels=['Intact\ntangent','Removed\ntangent','Removed\nfull addition'];colors=['#a08560','#28649c']
    for ax,metric,title in [(axs[0],'pooled','Energy-weighted actual − angle-control gap'),(axs[1],'site_mean','Equal-site actual − angle-control gap')]:
        for j,(p,group) in enumerate([(single,'One document, 32 rows'),(multi,'64 distinct documents')]):
            values=[];lo=[];hi=[]
            for b,g in conditions:
                v=p['contrasts'][f'{b}_{g}_{metric}'];values.append(v['gap']);lo.append(v['gap']-v['paired_row_bootstrap95'][0]);hi.append(v['paired_row_bootstrap95'][1]-v['gap'])
            ax.errorbar(np.arange(3)+(j-.5)*.12,values,yerr=[lo,hi],fmt='o',capsize=4,color=colors[j],label=group)
        ax.set_xticks(range(3),labels);ax.axhline(0,color='black',lw=.7);ax.set_ylabel('Rank-16 coverage difference');ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.2)
    ax=axs[2]
    for fam,color in [('actual','#28649c'),('shuffled','#b87529'),('angle','#8b5999')]:
        v=[multi['head_results'][f'native_removed_{fam}_add_h{h}']['16'] for h in range(9)];ax.plot(range(9),v,'o-',label=fam,color=color,markersize=4)
    ax.set_title('64 documents: source-removed full addition',fontsize=11);ax.set_xlabel('Source attention head');ax.set_ylabel('Rank-16 response coverage');ax.grid(alpha=.2);ax.legend(loc='best',fontsize=9)
    handles,leg=axs[0].get_legend_handles_labels();fig.legend(handles,leg,loc='lower center',bbox_to_anchor=(.34,.035),ncol=2)
    fig.suptitle('Replication changes the sampling unit from rows to documents')
    fig.text(.5,.007,'Intervals resample panel rows: within one document (brown), across 64 documents (blue). Fixed shams and one model.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.14,1,.94])
    for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_L8_DOCUMENT_REPLICATION.{ext}',dpi=170)
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
