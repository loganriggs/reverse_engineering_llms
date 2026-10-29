"""M77 fixed per-family operator comparison."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent

def main():
    rows=json.loads((P/'validation/native_input_operator_comparison.json').read_text())['summaries']
    families=['sequence_copy','weekday_successor','tense','continent_recall','object_function','spatial_inverse','role_binding'];labels=['Sequence\ncopy','Weekday\nsuccessor','Temporal\ncues','Continent\nrecall','Object\nfunction','Spatial\ninverse*','Role\nbinding*']
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained');x=np.arange(7)
    for ax,region,title in zip(axes,['changed_input_positions','final_position'],['Changed input positions','Final position']):
        for offset,predictor,label,color in [(-.18,'shared_J_at_training_mean','Shared training-mean operator','#0072b2'),(.18,'recipient_J','Recipient-local operator','#d55e00')]:
            values=[next(r['equal_family_response_error'] for r in rows if r['scope']=='family:'+f and r['region']==region and r['predictor']==predictor) for f in families]
            ax.bar(x+offset,values,width=.36,label=label,color=color)
        ax.axhline(1,color='gray',ls='--');ax.set(xticks=x,xticklabels=labels,ylabel='Relative causal response error',title=title,yscale='log');ax.grid(axis='y',alpha=.2)
    handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,loc='outside lower center',ncol=2)
    fig.suptitle('Native MLP1: context-local operators outperform the fixed mean operator\n*Fails native contrast competence gate; retained diagnostically. Exact midpoint control verified separately.')
    fig.savefig(P/'figures/native_input_operators.png',dpi=180);fig.savefig(P/'figures/native_input_operators.svg')
if __name__=='__main__':main()
