"""Exact signed-energy change for all three native-output final-position cases."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;V=P/'validation';d=json.loads((V/'product_error_interaction_whole_comparison.json').read_text());assert d['audit_sha256']==hashlib.sha256((V/'product_error_interaction_whole_audit.json').read_bytes()).hexdigest()
fig,axes=plt.subplots(1,3,figsize=(12,4.8),sharey=True)
for ax,method in zip(axes,['energy','random9001','random9002']):
 r=next(r for r in d['comparisons'] if r['space']=='native' and r['method']==method and r['scope']=='all' and r['region']=='final_position');t=r['terms']['family'];parts=t['delta_E2_decomposition'];vals=[parts[k] for k in ['isolated_norms','error_alignment','finite_interaction_net']];cum=0
 for i,val in enumerate(vals):
  ax.bar(i,val,bottom=cum,color=['#277b72','#c88632','#a95550'][i],width=.65);ax.text(i,cum+val+(0.025 if val>=0 else -.055),f'{val:+.3f}',ha='center',fontsize=9);cum+=val
 ax.bar(3,t['delta']['E2'],color='#333333',width=.65);ax.text(3,t['delta']['E2']+.025,f"{t['delta']['E2']:+.3f}",ha='center',fontsize=9)
 ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(4),['Isolated\nerrors','Error\nalignment','Net finite\ninteraction','Total']);ax.tick_params(axis='x',labelsize=9);ax.set_title(method);ax.grid(axis='y',alpha=.15)
axes[0].set_ylabel('Change in equal-family relative response error');axes[0].set_ylim(-.97,.24)
fig.suptitle('Smaller component errors, less cancellation, slightly worse combined fidelity\nNative output space · final position · all seven whole-task families · scaled minus raw',fontsize=12)
fig.text(.5,.015,'Positive net interaction change does not mean a larger interaction norm: the interaction norm decreases in all three cases.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.9));fig.savefig(P/'figures/product_error_interaction.png',dpi=160);fig.savefig(P/'figures/product_error_interaction.svg');plt.close(fig)
