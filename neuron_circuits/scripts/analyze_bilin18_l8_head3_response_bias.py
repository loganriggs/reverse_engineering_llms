#!/usr/bin/env python3
"""Post-result decomposition of payload-training gains into mean and varying errors."""
from pathlib import Path
import json,numpy as np,torch
from run_bilin18_l8_direct_objective_grid import response_data,qresponse
ROOT=Path(__file__).resolve().parents[1];STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_head3_payload_training'
@torch.no_grad()
def main():
 torch.set_num_threads(6)
 f=np.load(STEM.parent/'bilin18_l8_intervention_input_metric_frozen.npz');q=torch.from_numpy(f['exact_forms']).cuda().float();mean=torch.from_numpy(f['mean']).cuda().float()
 fitted=np.load(STEM.with_name(STEM.name+'_frozen.npz'));test=np.load(STEM.with_name(STEM.name+'_test.npz'));train=np.load(STEM.parent/'bilin18_l8_coverage_grid_train_validation.npz')
 tp={k:torch.from_numpy(test['input_'+k]).cuda() for k in ['native','drop3']};dp={k:torch.from_numpy(train['train_'+k]).cuda() for k in ['native','drop3']}
 d,m,lin,y=response_data(tp,['drop3'],q,mean);dt,mt,lt,yt=response_data(dp,['drop3'],q,mean)
 energy=float(y.square().sum());centered_energy=float((y-y.mean(0)).square().sum());N=len(y);results={}
 for name in ['baseline']+[f'{condition}_rep{r}' for r in [0,1] for condition in ['even_only','actual','shuffled','gaussian','isotropic']]:
  u=torch.from_numpy(fitted[name+'_u']).cuda();w=torch.from_numpy(fitted[name+'_w']).cuda()
  pred=lin+qresponse(d,m,u,w);train_pred=lt+qresponse(dt,mt,u,w)
  err=pred-y;bias=float(N*err.mean(0).square().sum());centered=float((err-err.mean(0)).square().sum())
  training_correction=(yt-train_pred).mean(0)
  results[name]={'fidelity':1-float(err.square().sum())/energy,'mean_bias_mse_over_target_energy':bias/energy,'centered_error_over_target_energy':centered/energy,
      'centered_response_fidelity':1-centered/centered_energy,'training_mean_corrected_fidelity':1-float((err+training_correction).square().sum())/energy,'mean_error_by_axis':err.mean(0).tolist()}
 contrasts={}
 for r in [0,1]:
  a=results[f'actual_rep{r}'];b=results[f'even_only_rep{r}'];gain=a['fidelity']-b['fidelity']
  contrasts[str(r)]={'total_gain':gain,'mean_bias_reduction':b['mean_bias_mse_over_target_energy']-a['mean_bias_mse_over_target_energy'],
    'centered_error_reduction':b['centered_error_over_target_energy']-a['centered_error_over_target_energy']}
 out={'scope':'Post-result bias decomposition. Training-mean correction is a source-declared diagnostic response offset, not an integrable decoder or a test-selected fit.',
      'target_mean_fraction':1-centered_energy/energy,'methods':results,'actual_vs_even':contrasts}
 STEM.with_name(STEM.name+'_bias_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
