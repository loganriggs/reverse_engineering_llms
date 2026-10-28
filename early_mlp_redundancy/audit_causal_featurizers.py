"""M8 independent NumPy encoder/group/union/decoder and causal-score audit."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent/'validation';torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def logsoft(x):
 z=x.astype(np.float64);z-=z.max(-1,keepdims=True);return z-np.log(np.exp(z).sum(-1,keepdims=True))
rp=P/'causal_featurizers.json';r=json.loads(rp.read_text());pp=P/'causal_featurizer_protocol.json';assert sha(pp)==r['protocol_sha256'];m4=json.loads((P/'natural_token_context.json').read_text());data=torch.load(m4['cache'],weights_only=True);Y=data['Y'][1].numpy();u=data['unique_indices'].numpy();docs=data['documents'].numpy();train=u[docs[u]<32];eval_y=Y[4096:];model_errors=[];metric_errors=[];partition_errors=[];boundary_records=[];unconditioned_errors=[]
for model in r['models']:
 p=Path(model['path']);assert sha(p)==model['sha256'];s=torch.load(p,weights_only=True);state={k:v.numpy() for k,v in s['state'].items()};mean=s['mean'].numpy();scale=float(s['scale']);assert np.allclose(mean,Y[train].mean(0),rtol=1e-5,atol=.02);assert abs(scale-np.sqrt(np.mean((Y[train]-mean)**2)))/scale<1e-6
 x=(eval_y.astype(np.float64)-mean)/scale;coords=x@state['encoder'].T+state['encoder_bias'];kind=s['kind'];M=state['membership'];q=np.zeros_like(coords)
 if kind=='scalar':q[np.arange(len(x))[:,None],np.argsort(np.abs(coords),axis=1)[:,-32:]]=1
 else:
  assert np.all((M==0)|(M==1))
  if kind=='block':assert np.all(M.sum(0)==1) and np.all(M.sum(1)==4)
  else:assert np.all(M.sum(0)==2) and np.all(M.sum(1)==8)
  energies=coords**2@M.T;g=np.zeros_like(energies);k=8 if kind=='block' else 4;g[np.arange(len(x))[:,None],np.argsort(energies,axis=1)[:,-k:]]=1;q=(g@M>0).astype(float)
 # Strict CUDA replay diagnoses discontinuous TopK rounding separately from decoder errors.
 m=Featurizer(s['kind'],s['seed']).cuda();m.load_state_dict(s['state'])
 with torch.no_grad():
  xx=(torch.from_numpy(eval_y).cuda()-s['mean'].cuda())/s['scale'].cuda();pprod,qprod,_=m(xx);production=(pprod*s['scale'].cuda()+s['mean'].cuda()).cpu();assert torch.equal(production,s['reconstructions']);qprod=qprod.cpu().numpy()
 changed=np.any(q!=qprod,axis=1);scores=np.abs(coords) if kind=='scalar' else coords**2@M.T;k=32 if kind=='scalar' else 8;ordered=np.sort(scores,axis=1);gaps=(ordered[:,-k]-ordered[:,-k-1])/np.maximum(np.abs(ordered[:,-k]),1e-12)
 assert np.all(gaps[changed]<1e-5)
 boundary_records.append(dict(kind=kind,seed=s['seed'],objective=s['objective'],different_support_rows=int(changed.sum()),relative_cutoff_gaps=gaps[changed].tolist(),cuda_checkpoint_replay_exact=True))
 D0=state['dictionary'].astype(np.float64);D0/=np.linalg.norm(D0,axis=0,keepdims=True);unconditioned=((coords*q)@D0.T+state['bias'])*scale+mean;unconditioned_errors.append(float(np.linalg.norm(unconditioned-s['reconstructions'].numpy())/np.linalg.norm(s['reconstructions'].numpy())))
 # For numerical ties only, audit decoder conditional on the recorded implementation's hard choice.
 q[changed]=qprod[changed];del m
 D=state['dictionary'].astype(np.float64);D/=np.linalg.norm(D,axis=0,keepdims=True);recon=((coords*q)@D.T+state['bias'])*scale+mean;cached=s['reconstructions'].numpy();err=float(np.linalg.norm(recon-cached)/np.linalg.norm(cached));model_errors.append(err);assert err<1e-4,(kind,err)
 row=s['record'];assert abs(q.sum(1).mean()-row['active_mean'])<1e-8;assert q.sum(1).max()==row['active_max'];error=cached.astype(np.float64)-eval_y;relative=np.sum(error**2)/np.sum(eval_y.astype(np.float64)**2);assert abs(relative-row['eval_relative_raw_error'])<1e-10
 token=data['tokens'][4096:].numpy();b=np.empty_like(error);tb=np.empty_like(error)
 for t in np.unique(token):b[token==t]=error[token==t].mean(0);tb[token==t]=eval_y[token==t].astype(np.float64).mean(0)
 for name,value in [('eval_token_mean_error_energy',np.sum(b*b)),('eval_within_token_error_energy',np.sum((error-b)**2)),('eval_within_token_error_relative_to_within_target',np.sum((error-b)**2)/np.sum((eval_y-tb)**2))]:
  delta=abs(value-row[name])/max(abs(value),1e-10);partition_errors.append(delta);assert delta<1e-10
sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True)
for s in samples:
 ref=s['native'].numpy();lp=logsoft(ref);ids=s['ids'].numpy();native_ce=float(-np.take_along_axis(lp[:,:-1],ids[:,1:,None],axis=-1).mean())
 for c in [s]:
  out=c['output'].numpy();lq=logsoft(out);row=next(x for x in r['causal_records'] if x['document']==s['document'] and x['kind']==c['kind'] and x['seed']==c['seed'] and x['objective']==c['objective']);ce=float(-np.take_along_axis(lq[:,:-1],ids[:,1:,None],axis=-1).mean());vals=dict(KL=float((np.exp(lp)*(lp-lq)).sum(-1).mean()),CE=ce,delta_CE=ce-native_ce,top1_agreement=float((out.argmax(-1)==ref.argmax(-1)).mean()))
  for k,v in vals.items():metric_errors.append(abs(v-row[k]));assert metric_errors[-1]<1e-8
for row in r['causal_summaries']:
 records=[x for x in r['causal_records'] if x['kind']==row['kind'] and x['seed']==row['seed'] and x['objective']==row['objective']];assert len(records)==32
 for k in ['KL','CE','delta_CE','top1_agreement']:assert abs(np.mean([x[k] for x in records])-row[k])<1e-10
out=dict(status='passed',model_forward_replays=8,eval_positions_per_model=4096,max_model_relative_error=max(model_errors),max_unconditioned_numpy_relative_error=max(unconditioned_errors),topk_boundary_records=boundary_records,max_partition_relative_error=max(partition_errors),independent_full_output_checks=8,max_causal_metric_absolute_error=max(metric_errors),all256_causal_record_aggregates_checked=True,result_sha256=sha(rp),scope='CUDA checkpoint replay exact. NumPy gate selection agrees except documented near-cutoff float32/float64 ties; tie rows use production support for conditional decoder audit. NumPy forward replay of all8learned featurizers, membership/union/cost semantics, token/context errors and8sample full-output metrics. No independent optimizer or full native-model implementation.')
(P/'causal_featurizers_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
