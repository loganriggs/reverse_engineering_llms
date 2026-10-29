"""Audited six-candidate retention and joint-response comparisons."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_pca_encoder import V

def main():
 natural=json.loads((V/'product_dictionary_natural_comparison.json').read_text())['summaries'];whole=json.loads((V/'product_dictionary_factorial_comparison.json').read_text())['summaries'];source=json.loads((V/'product_dictionary_source_factorial_comparison.json').read_text())['summaries']
 fig,axes=plt.subplots(2,2,figsize=(11,7.5));arms=['initial','intact_KL','joint_three_corner'];labels=['Initial','KL only','Joint causal']
 panels=[(natural,'mean_KL','Natural text: output KL','Mean KL'),(whole,'source_reconstructed_reconstructed_family','Whole tasks: source-position replacement','Equal-family response error'),(whole,'final_reconstructed_reconstructed_family','Whole tasks: final-position replacement','Equal-family response error'),(source,'final_reconstructed_reconstructed_family','Source-group holdout: final-position replacement','Equal-family response error')]
 for ax,(data,key,title,ylabel) in zip(axes.flat,panels):
  for seed,color in [(71,'#2176ae'),(72,'#d16b29')]:
   values=[next(x[key] for x in data if x['seed']==seed and x['arm']==arm and x.get('scope','all')=='all') for arm in arms]
   ax.plot(range(3),values,'o-',color=color,label=f'Seed {seed}',linewidth=1.8)
  ax.set_xticks(range(3),labels);ax.set_ylabel(ylabel);ax.set_title(title,fontsize=11);ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
 axes[0,0].legend(frameon=False);fig.suptitle('Shared native products: small, site-dependent supervision effects',fontsize=14)
 fig.text(.5,.015,'Lower is better. Both predicted background and change for causal panels. Different vertical scales; all candidates retained.',ha='center',fontsize=9)
 fig.tight_layout(rect=[0,.04,1,.95]);dest=V.parent/'figures';fig.savefig(dest/'product_dictionary_transfer.png',dpi=170);fig.savefig(dest/'product_dictionary_transfer.svg');plt.close(fig)
if __name__=='__main__':main()
