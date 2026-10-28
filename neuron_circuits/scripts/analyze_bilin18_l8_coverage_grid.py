import sys,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/home/loganriggs/Coding/neuron_circuits/scripts')
from run_bilin18_l8_direct_objective_grid import qresponse,response_data
root=Path('/home/loganriggs/Coding/neuron_circuits');art=root/'artifacts/bilin18_tensor_decomposition';stem=art/'bilin18_l8_coverage_grid';a=json.loads(stem.with_suffix('.json').read_text())
rows=np.load(stem.with_name(stem.name+'_row_errors.npz'));ix=np.random.default_rng(20261030).integers(128,size=(2000,128))
e=rows['odd_energy'];diff=rows['rows512_even_odd_error']-rows['rows512_all_odd_error']-rows['rows128_even_odd_error']+rows['rows128_all_odd_error']
interaction={'odd_fidelity_difference_in_differences':float(diff.sum()/e.sum()),'paired_row_bootstrap_95':np.quantile(diff[ix].sum(1)/e[ix].sum(1),[.025,.975]).tolist()}
frozen=np.load(art/'bilin18_l8_intervention_input_metric_frozen.npz');q=torch.from_numpy(frozen['exact_forms']).cuda().float();mean=torch.from_numpy(frozen['mean']).cuda().float()
factors=np.load(stem.with_name(stem.name+'_frozen.npz'));test=np.load(stem.with_name(stem.name+'_test.npz'));panel={k.removeprefix('input_'):torch.from_numpy(test[k]).cuda().float() for k in test.files if k.startswith('input_')}
methods=['baseline','value_only','rows128_even','rows128_all','rows512_even','rows512_all'];headwise={};axiswise={}
with torch.no_grad():
 for h in range(9):
  d,m,lin,target=response_data(panel,[f'drop{h}'],q,mean);energy=target.square();headwise[str(h)]={'energy':float(energy.sum()),'methods':{}}
  for name in methods:
   u=torch.from_numpy(factors[name+'_u']).cuda();w=torch.from_numpy(factors[name+'_w']).cuda()
   error=(lin+qresponse(d,m,u,w)-target).square()
   headwise[str(h)]['methods'][name]={'fidelity':float(1-error.sum()/energy.sum()),'per_axis_fidelity':(1-error.sum(0)/energy.sum(0)).tolist()}
 for name in methods:
  u=torch.from_numpy(factors[name+'_u']).cuda();w=torch.from_numpy(factors[name+'_w']).cuda()
  d,m,lin,target=response_data(panel,[f'drop{h}' for h in [1,3,5,7]],q,mean)
  axiswise[name]=(1-(lin+qresponse(d,m,u,w)-target).square().sum(0)/target.square().sum(0)).tolist()
den=sum(headwise[str(h)]['energy'] for h in [1,3,5,7]);contributions={}
for h in [1,3,5,7]:
 r=headwise[str(h)];share=r['energy']/den;gain=r['methods']['rows512_all']['fidelity']-r['methods']['rows512_even']['fidelity']
 contributions[str(h)]={'response_energy_share':share,'contribution_to_pooled_gain':share*gain}
total=sum(r['contribution_to_pooled_gain'] for r in contributions.values())
for r in contributions.values():r['share_of_pooled_gain']=r['contribution_to_pooled_gain']/total
out={'head_contributions_to_source_gain':contributions,'scope':'Post-result diagnostics, not additional preregistered discoveries','interaction':interaction,'headwise':headwise,'odd_axiswise':axiswise};stem.with_name(stem.name+'_diagnostics.json').write_text(json.dumps(out,indent=2))
print('interaction',interaction)
for h,r in headwise.items():print('head',h,{n:round(v['fidelity'],3) for n,v in r['methods'].items()})
print('axis',axiswise)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(11.5,4.4))
for source,label,color in [('even','Train on five even heads','#3267a8'),('all','Train on all nine heads','#b64b36')]:
 vals=[a['test'][f'rows{n}_{source}']['groups']['odd'] for n in [128,512]]
 axs[0].plot([0,1],vals,'o-',label=label,color=color,lw=2)
 for x,v in enumerate(vals):axs[0].annotate(f'{v:.3f}',(x,v),xytext=(0,7),textcoords='offset points',ha='center',color=color)
axs[0].set_xticks([0,1],['128 rows','512 rows']);axs[0].set_ylim(.55,.79);axs[0].set_ylabel('Response fidelity on odd heads');axs[0].set_title('Source coverage makes more text useful',loc='left',fontweight='bold');axs[0].legend(frameon=False,loc='upper left',fontsize=9)
for i,(name,label,color) in enumerate([('baseline','Spectral','#999999'),('value_only','Value only','#7b689d'),('rows512_even','Five heads','#3267a8'),('rows512_all','Nine heads','#b64b36')]):
 vals=[a['test'][name]['groups'][g] for g in ['odd','gaussian','isotropic']]
 axs[1].bar(np.arange(3)+(i-1.5)*.19,vals,width=.18,label=label,color=color)
axs[1].set_xticks(range(3),['Actual odd-head','Gaussian in head span','Isotropic']);axs[1].set_ylim(0,.85);axs[1].set_ylabel('Response fidelity');axs[1].set_title('Better actual responses remain specialized',loc='left',fontweight='bold');axs[1].legend(frameon=False,ncol=2,fontsize=9)
fig.text(.01,.04,'Head 3 accounts for 86% of the pooled source-coverage gain at 512 rows; benefits are not uniform across heads.',fontsize=9)
fig.text(.01,.01,'Fixed rank 64 per scalar; four L8 coordinates; new test text. Odd heads are unseen sources only in five-head fits.',fontsize=9)
fig.tight_layout(rect=[0,.10,1,1]);fig.savefig(art/'BILIN18_L8_COVERAGE_GRID.png',dpi=180);fig.savefig(art/'BILIN18_L8_COVERAGE_GRID.svg')
