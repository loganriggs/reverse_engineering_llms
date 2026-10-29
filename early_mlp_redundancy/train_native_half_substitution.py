"""M42 training-only low-rank native-half substitution, no task data fitting."""
from pathlib import Path
import json,hashlib
import torch
from inspect_model import load_model,trace
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fit(x,y,rank,ridge_fraction):
 x=x.double();y=y.double();mx=x.mean(0);my=y.mean(0);xc=x-mx;yc=y-my;cov=xc.T@xc;e,U=torch.linalg.eigh(cov);U=U[:,-rank:];z=xc@U;gram=z.T@z;lam=ridge_fraction*gram.trace()/rank;B=torch.linalg.solve(gram+lam*torch.eye(rank,dtype=z.dtype),z.T@yc)
 return dict(input_mean=mx,target_mean=my,U=U,B=B,ridge=lam,eigenvalues=e[-rank:],training_mean_error=yc.square().sum(),training_map_error=(z@B-yc).square().sum(),eigen_gap=e[-rank]-e[-rank-1])
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 pp=V/'native_half_substitution_protocol.json';q=json.loads(pp.read_text());assert q['source_sha256']==sha(Path(__file__));cp=V/'natural_token_context_cohort.json';assert q['cohort_sha256']==sha(cp);c=json.loads(cp.read_text());dp=V/'mlp1_group_deletion.json';assert q['masks_sha256']==sha(dp);masks=json.loads(dp.read_text())['removed_sets'];assert not (V/'native_half_substitution_training.json').exists()
 assert not set(x['document_hash'] for x in c['records'][:32])&set(x['document_hash'] for x in c['records'][32:])
 cache=A/'emlp_m42_train_halves.pt';calls=0
 if cache.exists():
  data=torch.load(cache,weights_only=True);assert data['protocol_sha256']==sha(pp);receipt=data['checkpoint_sha256'];halves=data['halves'];calls=data['forward_count']
 else:
  model,rec=load_model();device=next(model.parameters()).device;halves={};receipt=rec.weights_sha256
  for layer in [0,1,2]:
   for half in ['half_01','half_23']:halves[f'{layer}:{half}']=[]
  for doc in range(32):
   ids=torch.tensor([c['token_ids'][doc]],device=device);_,cap=trace(model,ids);calls+=1
   for layer in [0,1,2]:
    mlp=model.transformer.h[layer].mlp;prod=cap[layer]['products']
    for half in ['half_01','half_23']:
     mask=torch.zeros(4608,device=device);mask[masks[half]]=1;halves[f'{layer}:{half}'].append(mlp.Down(prod*mask)[0].cpu())
   if doc%8==7:print('TRAIN_DOCUMENTS',doc+1,flush=True)
  halves={k:torch.cat(v) for k,v in halves.items()};torch.save(dict(protocol_sha256=sha(pp),checkpoint_sha256=receipt,halves=halves,forward_count=calls),cache)
 records=[];dest=A/'emlp_m42_maps';dest.mkdir(exist_ok=True)
 for layer in [0,1,2]:
  for removed in ['half_01','half_23']:
   retained='half_23' if removed=='half_01' else 'half_01';x=halves[f'{layer}:{retained}'];y=halves[f'{layer}:{removed}'];assert x.shape==y.shape==(4096,1152);state=fit(x,y,q['rank'],q['ridge_fraction']);state.update(layer=layer,removed=removed,retained=retained,protocol_sha256=sha(pp));path=dest/f'{layer}_{removed}.pt';torch.save(state,path);records.append(dict(layer=layer,removed=removed,retained=retained,path=str(path),sha256=sha(path),train_rows=4096,rank=q['rank'],ridge=float(state['ridge']),training_mean_error=float(state['training_mean_error']),training_map_error=float(state['training_map_error']),map_mean_ratio=float(state['training_map_error']/state['training_mean_error']),eigen_gap=float(state['eigen_gap'])));print('FIT',layer,removed,records[-1]['map_mean_ratio'],flush=True)
 out=dict(protocol_sha256=sha(pp),checkpoint_sha256=receipt,cache_path=str(cache),cache_sha256=sha(cache),forward_count=calls,models=records,scope='Fit on4096 positions of natural documents0–31 only; rank32 and ridge fraction.001 fixed before fitting. No evaluation or task fitting; training error alone not transfer or causal evidence.')
 (V/'native_half_substitution_training.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
