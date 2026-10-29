"""M96 audited fixed-run training targets and natural-output movement."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_pca_encoder import V

def main():
 t=json.loads((V/'product_dictionary_training_targets_comparison.json').read_text());m=json.loads((V/'product_dictionary_movement_comparison.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 for seed,color in [(71,'#2176ae'),(72,'#d16b29')]:
  rows={r['arm']:r for r in t['summaries'] if r['seed']==seed and r['family'] is None};axes[0].plot(range(3),[rows['intact_KL']['B_initial'],rows['intact_KL']['B_final'],rows['joint_three_corner']['B_final']],'o-',color=color,label=f'Seed {seed}')
  vals=[next(100*r['relative_write_movement'] for r in m['summaries'] if r['seed']==seed and r['arm']==arm and r['population']=='evaluation') for arm in ['intact_KL','joint_three_corner']];axes[1].plot(range(2),vals,'o-',color=color,label=f'Seed {seed}')
 axes[0].set_xticks(range(3),['Initial','KL only','Joint causal']);axes[0].set_ylabel('Balanced causal training error B');axes[0].set_title('Real improvement on supervised targets')
 axes[1].set_xticks(range(2),['KL only','Joint causal']);axes[1].set_ylabel('Write-change norm / native-write norm (%)');axes[1].set_title('Small movement on natural evaluation text')
 for ax in axes:ax.grid(alpha=.2);ax.legend(frameon=False);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
 fig.suptitle('Learning occurred; broad causal transfer did not follow',fontsize=14);fig.text(.5,.015,'Different datasets and metrics. Training schedules retain repeats; output movement pools all32natural evaluation documents.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.92]);fig.savefig(V.parent/'figures/product_dictionary_diagnosis.png',dpi=170);fig.savefig(V.parent/'figures/product_dictionary_diagnosis.svg');plt.close(fig)
if __name__=='__main__':main()
