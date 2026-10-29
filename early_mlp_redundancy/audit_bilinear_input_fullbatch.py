"""Independent final-state/schedule and full-training reconstruction audit."""
import hashlib,json
import numpy as np
import torch
from fit_pca_encoder import sha,V

def main():
 torch.set_num_threads(1);rp=V/'bilinear_input_fullbatch_results.json';r=json.loads(rp.read_text());assert r['receipt']['source_sha256']==sha(V.parent/'train_bilinear_input_fullbatch.py');initials=json.loads((V/'bilinear_input_initial_states.json').read_text())['models'];ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];cache=torch.load(ref['path'],weights_only=True);m4=json.loads((V/'natural_token_context.json').read_text());assert sha(m4['cache'])==m4['cache_sha256'];w=torch.load(m4['cache'],weights_only=True)['Y'][1][cache['indices']].double().numpy();z=cache['z'][cache['indices']].double().numpy();rows=[]
 for entry in r['models']:
  assert sha(entry['path'])==entry['sha256'];s=torch.load(entry['path'],weights_only=True);start=next(e for e in initials if e['kind']==entry['kind'] and e['seed']==entry['seed']);assert s['initial_sha256']==start['sha256'];assert sha(start['path'])==start['sha256'];init=torch.load(start['path'],weights_only=True);assert s['optimizer_updates']==len(s['history'])==1000 and np.isfinite(s['history']).all();state=s['state'];assert tuple(s['optimizer_defaults']['betas'])==(0.9,0.999);assert s['optimizer_defaults']['lr']==.001 and s['optimizer_defaults']['eps']==1e-8
  for k,v in init['statistics'].items():assert torch.equal(state[k],v)
  assert torch.equal(state['features.membership'],init['state']['membership']);assert all(torch.isfinite(t).all() for t in state.values());assert torch.allclose(state['features.dictionary'].norm(dim=0),torch.ones(128),rtol=1e-5,atol=1e-6)
  h=hashlib.sha256()
  for _ in range(1000):h.update(torch.arange(4092).numpy().tobytes())
  assert h.hexdigest()==s['schedule_sha256'];progress=torch.load(entry['path'].replace('.pt','.progress.pt'),weights_only=True);assert progress['receipt']==s['receipt'] and len(progress['history'])==1000;assert all(int(v['step'])==1000 for v in progress['optimizer']['state'].values())
  for k,v in state.items():assert torch.equal(v,progress['state'][k].cpu())
  d={k:t.double().numpy() for k,t in state.items()};x=(z-d['input_mean'])/d['input_scale'];a=(x@d['features.left'].T+d['features.left_bias'])*(x@d['features.right'].T+d['features.right_bias']);M=d['features.membership'];kind=entry['kind'];q=np.ones_like(a)
  if kind!='dense':
   score=np.abs(a) if kind=='scalar' else np.square(a)@M.T;k=32 if kind=='scalar' else (8 if kind=='block' else 4);indices=np.argsort(-score,axis=1,kind='stable')[:,:k];g=np.zeros_like(score);np.put_along_axis(g,indices,1,axis=1);q=g if kind=='scalar' else (g@M>0)
  D=d['features.dictionary'];D=D/np.sqrt(np.square(D).sum(0));pred=((a*q)@D.T+d['features.bias'])*d['output_scale']+d['output_mean'];mse=float(np.mean(np.square((pred-w)/d['output_scale'])));active=float(q.sum(1).mean());assert np.isclose(mse,s['full_training_MSE'],rtol=1e-5,atol=1e-7),(kind,entry['seed'],mse,s['full_training_MSE']);assert np.isclose(active,s['active_mean'],rtol=0,atol=1/4092)
  rows.append(dict(kind=kind,seed=entry['seed'],training_MSE_independent=mse,training_MSE_recorded=s['full_training_MSE'],active_mean=active,optimizer_updates=1000,schedule_sha256=s['schedule_sha256']));print('AUDITED',kind,entry['seed'],flush=True)
 assert len(rows)==8;out=dict(status='passed',source_sha256=sha(__file__),result_sha256=sha(rp),models=rows,total_optimizer_updates=8000,scope='Independent perstep minibatch schedule hash, savedAdamsteps/finalstates/buffers, and NumPy fulltraining predictions/gates/MSE. No independent optimizertrajectory replay and no heldoutquality claim.')
 path=V/'bilinear_input_fullbatch_audit.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
