"""All fixed rank/basis matched curves; no selected winner."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from fit_pca_encoder import V,sha

def main():
 r=json.loads((V/'native_mixed_interaction_comparison.json').read_text())
 for name,h in r['inputs'].items():assert sha(V/name)==h
 lookup={(x['cohort'],x['arm'],x['scope']):x for x in r['rows']};bases=[('euclidean','Input PCA','#2563eb'),('isotropic_operator','Operator PCA','#d97706'),('training_direction_operator','Direction-weighted PCA','#059669')];panels=[('natural','mean_KL','Natural-text output KL'),('whole','source_reconstructed_reconstructed_family','Whole-task source response'),('whole','final_reconstructed_reconstructed_family','Whole-task final response'),('natural','local_relative_error','Natural local-write error'),('source','source_reconstructed_reconstructed_family','Source-group source response'),('source','final_reconstructed_reconstructed_family','Source-group final response')];fig,axes=plt.subplots(2,3,figsize=(12,7.5))
 for ax,(cohort,metric,title) in zip(axes.flat,panels):
  for basis,label,color in bases:
   for key,style in [('restricted','--'),('mixed','-')]:
    values=[lookup[cohort,basis+'__'+str(k),'all']['metrics'][metric][key] for k in [4,16,64]];ax.plot([4,16,64],values,style,color=color,marker='o',markersize=4,linewidth=1.8)
  if cohort!='natural':ax.axhline(1,color='#9ca3af',linewidth=.8,zorder=0)
  ax.set_xscale('log',base=4);ax.set_xticks([4,16,64],labels=['4','16','64']);ax.set_yscale('log');ax.set_title(title,fontsize=11);ax.set_xlabel('Context coordinates');ax.set_ylabel('Error (lower is better)');ax.grid(alpha=.15)
 handles=[Line2D([0],[0],color=c,label=l,linewidth=2) for _,l,c in bases]+[Line2D([0],[0],color='black',linestyle='--',label='Restricted quadratic'),Line2D([0],[0],color='black',label='Mixed interactions')]
 fig.suptitle('Retaining mixed interactions: large gains, with low-rank source exceptions',fontsize=14,y=.99);fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.95),ncol=5,frameon=False,fontsize=9);fig.tight_layout(rect=(0,0,1,.89));dest=V.parent/'figures';dest.mkdir(exist_ok=True)
 for suffix in ['png','svg']:fig.savefig(dest/('native_mixed_interactions.'+suffix),dpi=170,bbox_inches='tight')
 print(dest/'native_mixed_interactions.png')
if __name__=='__main__':main()
