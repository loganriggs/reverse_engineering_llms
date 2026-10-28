"""H1011: predict subject-number causal edges from upstream source states only."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_causal_source_prediction'
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
import bilin18_observed_model_facade as facade
import torch
out=A/'causal_source_prediction';out.mkdir(exist_ok=True)
config={'experiment':'H1011','training':'32 original discovery rows only; PCA, scaling and readouts fit only there','features':'subject-position five clean native source ports and their five intervention deltas, plus clean residual at same site','models':'mean; 15 source norm/cosine features; train-PCA4/8 linear and full quadratic, ridge0.1 fixed','scope':'Conditional upstream-state-to-causal-effect prediction, not token-only computation or trained BSF. Historical evaluation reused, no new OOD claim. Nine effects predicted separately, sum compared with full joint damage. Native prefix supplies sources; suffix never supplied as student input.'}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n')
load=facade.load_bilin18
facade.load_bilin18=lambda **kw:load(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),**kw)
runner.OUT=out/'native_replay.json'
old_eval=runner.component.evaluate;old_capture=runner.component._capture
batches=[];ports=[];pos=None

def capture(*args):
 result=old_capture(*args);raw,_,_,pp,_=result
 ix=torch.arange(len(raw),device=raw.device)
 ports.append((torch.stack([p[ix,pos] for p in pp],1).cpu().double().numpy(),raw[ix,pos].cpu().double().numpy()))
 return result

def evaluate(*args,**kwargs):
 global pos,ports
 pos=args[2];ports=[]
 result=old_eval(*args,**kwargs)
 assert len(ports)==2
 clean=ports[0][0];delta=ports[1][0]-clean
 batches.append(dict(tokens=args[1].cpu().numpy(),positions=pos.cpu().numpy(),clean=clean,delta=delta,raw=ports[0][1],atoms=result['atoms'].cpu().numpy().T,target=result['target'].cpu().numpy()))
 return result
runner.component._capture=capture;runner.component.evaluate=evaluate
runner.main()
# Resolve row identities explicitly instead of assuming batch order equals authority order.
original=runner.original_authority.build_rows();fresh=runner.fresh_authority.build_rows()
lookup={tuple(r['token_ids']):('discovery' if r['pair_index']<8 and r['template_id'] in ('near','behind') else 'context_ood' if r['pair_index']<8 else 'lexical_ood' if r['template_id'] in ('near','behind') else 'joint_ood') for r in original}
meta=[]
for bi,b in enumerate(batches):
 for ids in b['tokens']:meta.append({'split':lookup[tuple(ids)] if bi<2 else 'historical_fresh','token_sha256':hashlib.sha256(ids.tobytes()).hexdigest()})
clean=np.concatenate([b['clean'] for b in batches]);delta=np.concatenate([b['delta'] for b in batches]);raw=np.concatenate([b['raw'] for b in batches]);Y=np.concatenate([b['atoms'] for b in batches]);target=np.concatenate([b['target'] for b in batches]);X=np.concatenate([clean.reshape(160,-1),delta.reshape(160,-1),raw],1)
train=np.array([i for i,m in enumerate(meta) if m['split']=='discovery']);assert len(train)==32
assert len({m['token_sha256'] for m in meta})==160
xm=X[train].mean(0);u,sv,vh=np.linalg.svd(X[train]-xm,full_matrices=False);ym=Y[train].mean(0)
norms=np.concatenate([np.linalg.norm(clean,axis=2),np.linalg.norm(delta,axis=2),np.sum(clean*delta,axis=2)/(np.linalg.norm(clean,axis=2)*np.linalg.norm(delta,axis=2)).clip(1e-12)],1)
features={'norms15':norms};predictions={'mean':np.broadcast_to(ym,Y.shape).copy()};models={};residuals=[]
for rank in [4,8]:
 z=(X-xm)@vh[:rank].T;features[f'pca{rank}_linear']=z
 z=z/np.std(z[train],axis=0).clip(1e-12);ii,jj=np.triu_indices(rank);features[f'pca{rank}_quadratic']=np.concatenate([z,z[:,ii]*z[:,jj]],axis=1)
for name,f in features.items():
 fm=f[train].mean(0);fs=f[train].std(0).clip(1e-12);h=(f-fm)/fs;ht=h[train];gram=ht.T@ht/32+.1*np.eye(h.shape[1]);rhs=ht.T@(Y[train]-ym)/32;w=np.linalg.solve(gram,rhs);predictions[name]=ym+h@w;models[name]=(fm,fs,w);residuals.append(float(np.linalg.norm(gram@w-rhs)/np.linalg.norm(rhs)))
summary=[]
for split in sorted({m['split'] for m in meta}):
 ix=[i for i,m in enumerate(meta) if m['split']==split]
 for name,p in predictions.items():
  summary.append({'split':split,'model':name,'n':len(ix),'joint_relative_l2':float(np.linalg.norm(p[ix].sum(1)-target[ix])/np.linalg.norm(target[ix])),'edge_relative_l2':(np.linalg.norm(p[ix]-Y[ix],axis=0)/np.linalg.norm(Y[ix],axis=0)).tolist(),'pooled_edges_relative_l2':float(np.linalg.norm(p[ix]-Y[ix])/np.linalg.norm(Y[ix]))})
arrays=dict(X=X,Y=Y,target=target,train=train,clean=clean,delta=delta,raw=raw,pca_mean=xm,pca_basis=vh[:8],output_mean=ym)
for name,p in predictions.items():arrays['prediction_'+name]=p
for name,f in features.items():
 arrays['features_'+name]=f
 for key,val in zip(['mean','scale','weights'],models[name]):arrays[name+'_'+key]=val
np.savez(out/'captures.npz',**arrays);(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
result={'config':config,'summary':summary,'audits':{'train_rows':32,'total_rows':160,'duplicate_token_sequences':0,'normal_equation_relative_residual':max(residuals),'finite':bool(all(np.isfinite(p).all() for p in predictions.values())),'native_replay_all_gates':all(json.loads(runner.OUT.read_text())['predictions'].values()),'student_suffix_inputs':0,'new_reserved_test_examples':0},'terms':list(runner.component.TERMS)}
assert result['audits']['finite'] and result['audits']['native_replay_all_gates']
(A/f'{S}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
