"""M44 training-only token controls for six frozen native-half maps."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1)
 pp=V/'native_half_token_protocol.json';q=json.loads(pp.read_text());assert q['source_sha256']==sha(Path(__file__));tp=V/'native_half_substitution_training.json';assert sha(tp)==q['training_sha256'];tr=json.loads(tp.read_text());assert json.loads((V/'native_half_substitution_training_audit.json').read_text())['result_sha256']==sha(tp);cp=V/'natural_token_context_cohort.json';assert sha(cp)==q['cohort_sha256'];c=json.loads(cp.read_text());cache=Path(tr['cache_path']);assert sha(cache)==tr['cache_sha256'];data=torch.load(cache,weights_only=True);tokens=torch.tensor(c['token_ids'][:32]).flatten();assert tokens.numel()==4096;keys,inverse,counts=torch.unique(tokens,sorted=True,return_inverse=True,return_counts=True)
 tables={};checks=[]
 for k,h in data['halves'].items():
  sums=torch.zeros(len(keys),1152,dtype=torch.float64);sums.index_add_(0,inverse,h.double());means=sums/counts[:,None];tables[k]=means
  # Independent NumPy grouping, avoiding the scatter implementation.
  raw=h.numpy().astype(float);t=tokens.numpy();ind=np.stack([raw[t==int(tok)].mean(0) for tok in keys]);err=np.max(np.abs(ind-means.numpy()));assert err<1e-8
  centered=h.double()-h.double().mean(0);within=h.double()-means[inverse];checks.append(dict(key=k,mean_max_error=float(err),training_within_token_fraction=float(within.square().sum()/centered.square().sum())))
 dest=A/'emlp_m44_token_controls';dest.mkdir(exist_ok=True);models=[]
 for entry in tr['models']:
  src=Path(entry['path']);assert sha(src)==entry['sha256'];s=torch.load(src,weights_only=True);new=dict(s);new.update(token_ids=keys,token_counts=counts,token_input_mean=tables[f"{entry['layer']}:{entry['retained']}"],token_target_mean=tables[f"{entry['layer']}:{entry['removed']}"],token_protocol_sha256=sha(pp));path=dest/f"{entry['layer']}_{entry['removed']}.pt";torch.save(new,path);loaded=torch.load(path,weights_only=True)
  for k,value in s.items():
   if isinstance(value,torch.Tensor):assert torch.equal(value,loaded[k])
   else:assert value==loaded[k]
  models.append(dict(layer=entry['layer'],removed=entry['removed'],retained=entry['retained'],path=str(path),sha256=sha(path),source_sha256=entry['sha256']))
 # Coverage alone uses evaluation token IDs; never evaluation activations or outcomes.
 eval_tokens=torch.tensor(c['token_ids'][32:]).flatten();coverage=float(torch.isin(eval_tokens,keys).double().mean())
 out=dict(protocol_sha256=sha(pp),training_sha256=sha(tp),cache_sha256=sha(cache),models=models,training_positions=4096,token_types=len(keys),singleton_types=int((counts==1).sum()),natural_eval_token_coverage=coverage,scope='Training token means only; evaluation token IDs used solely to report coverage. All original mapping tensors unchanged.')
 rp=V/'native_half_token_controls.json';assert not rp.exists();rp.write_text(json.dumps(out,indent=2)+'\n');audit=dict(status='passed',result_sha256=sha(rp),checks=checks,all_original_map_tensors_bitwise_frozen=True,total_token_counts=int(counts.sum()),scope='Independent NumPy per-token means from all training half writes; all six original maps bitwise preserved. No causal transfer evaluated.');(V/'native_half_token_controls_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(dict(training_positions=4096,token_types=len(keys),singleton_types=int((counts==1).sum()),natural_eval_token_coverage=coverage,checks=checks),indent=2))
if __name__=='__main__':main()
