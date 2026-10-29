"""Native-space final-position factorial results; all three frozen rankings."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;V=P/'validation'
d=json.loads((V/'product_subset_factorial_comparison.json').read_text());assert d['audit_sha256']==hashlib.sha256((V/'product_subset_factorial_audit.json').read_bytes()).hexdigest()
fig,axes=plt.subplots(1,3,figsize=(12,4.6),sharey=True)
keys=['final_reconstructed_native_family','final_native_reconstructed_family','final_reconstructed_reconstructed_family'];labels=['Background\nonly','Change\nonly','Both'];colors=['#2876a0','#2876a0','#b44d2b']
for ax,method in zip(axes,['energy','random9001','random9002']):
 r=next(x for x in d['comparisons'] if x['scope']=='all' and x['space']=='native' and x['method']==method)
 for i,(k,c) in enumerate(zip(keys,colors)):
  raw=r['metrics'][k]['raw'];sc=r['metrics'][k]['scaled'];ax.scatter(i,raw,facecolors='white',edgecolors=c,s=55,zorder=3);ax.scatter(i,sc,color=c,s=40,zorder=3);ax.annotate('',xy=(i,sc),xytext=(i,raw),arrowprops=dict(arrowstyle='->',color=c,lw=1.6))
 ax.axhline(1,color='gray',ls='--',lw=1);ax.axhline(0,color='black',ls=':',lw=1);ax.set_xticks(range(3),labels,fontsize=9);ax.set_xlim(-.45,2.45);ax.set_title(method);ax.grid(axis='y',alpha=.15)
axes[0].set_ylabel('Equal-family relative causal-response error');axes[0].set_ylim(-.05,1.55)
fig.suptitle('Rescaling improves isolated substitutions but worsens their combination\nNative output space · final position · all seven whole-task families · raw (open) → scaled (filled)',fontsize=12)
fig.text(.5,.015,'Dashed line: zero-response baseline (error 1). Dotted line: zero error; native full controls are at numerical precision.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.88));fig.savefig(P/'figures/product_subset_causal.png',dpi=160);fig.savefig(P/'figures/product_subset_causal.svg');plt.close(fig)
