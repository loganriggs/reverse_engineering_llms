"""H313 disjoint marginal scalar calibration and oracle directional bound."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_neighbor_scaling'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for name,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/name).read_bytes()).hexdigest()==digest
 Y=torch.tensor(np.load(ART/'bilin18_l17_document_repartition_responses.npy'),device='cuda').reshape(18,4096,1152).double();idx=np.load(ART/'bilin18_l17_context_neighbors_indices.npz');rows={}
 for g in range(18):
  ref=Y[g,:3072];target=Y[g,3072:];ix=torch.tensor(idx[f'g{g}_joint'],device='cuda');chosen=ref[ix];unitmean=torch.nn.functional.normalize(chosen,dim=-1).mean(1);unitdir=torch.nn.functional.normalize(unitmean,dim=-1);preds={'raw_mean':chosen.mean(1),'unit_direction_mean_norm':unitdir*chosen.norm(dim=-1).mean(1)[:,None]};energy=target.square().sum(-1);rows[f'g{g}_energy']=energy.reshape(256,4).sum(1).cpu().numpy()
  for name,pred in preds.items():
   dot=(pred*target).sum(-1);norm2=pred.square().sum(-1);assert float(norm2.min())>0
   for metric,values in [('dot',dot),('norm2',norm2),('oracle',dot.clamp_min(0).square()/norm2),('cosine',dot/(norm2*energy).sqrt())]:rows[f'g{g}_{name}_{metric}']=values.reshape(256,4).sum(1).cpu().numpy()
 summary={};scalars={};boot=np.random.default_rng(cfg['seed']).integers(128,size=(2000,128))
 for family,groups in [('actual',range(9)),('angle',range(9,18))]:
  energy=sum(rows[f'g{g}_energy'] for g in groups);summary[family]={};scalars[family]={}
  for name in ['raw_mean','unit_direction_mean_norm']:
   dot=sum(rows[f'g{g}_{name}_dot'] for g in groups);norm2=sum(rows[f'g{g}_{name}_norm2'] for g in groups);oracle=sum(rows[f'g{g}_{name}_oracle'] for g in groups);alpha=max(0.,float(dot[:128].sum()/norm2[:128].sum()));scalars[family][name]=alpha;testE=energy[128:];gain=(2*alpha*dot-alpha**2*norm2)[128:];unscaled=(2*dot-norm2)[128:];summary[family][name]={'alpha':alpha,'unscaled_fidelity':float(unscaled.sum()/testE.sum()),'scaled_fidelity':float(gain.sum()/testE.sum()),'scaled95':np.quantile(gain[boot].sum(1)/testE[boot].sum(1),[.025,.975]).tolist(),'oracle_nonnegative_amplitude_fidelity':float(oracle[128:].sum()/testE.sum()),'mean_signed_cosine':float(sum(rows[f'g{g}_{name}_cosine'][128:].sum() for g in groups)/(128*4*9))}
 gates={'scaled10':all(summary[f]['unit_direction_mean_norm']['scaled_fidelity']>=.1 for f in summary),'direction_bound50':all(summary[f]['unit_direction_mean_norm']['oracle_nonnegative_amplitude_fidelity']>=.5 for f in summary)};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'scalars':scalars,'summary':summary,'gates':gates},indent=2)+'\n');print('FINISHED',summary,gates,flush=True)
if __name__=='__main__':main()
