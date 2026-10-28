"""H1012: frozen family lexical expansion and prospective noun-disjoint test."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_expanded_causal_prediction';out=A/'expanded_causal_prediction';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
import bilin18_observed_model_facade as facade
panel_path=A/'bilin18_expanded_causal_panel.json';panel=json.loads(panel_path.read_text());rows=panel['rows'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
config={'experiment':'H1012','panel_sha256':sha(panel_path),'training':'Original32 discovery plus192 expanded discovery; separate24training and8test noun pairs','models':'Unchanged H1011 mean,norms15,PCA4/8 linear/quadratic; ridge0.1; no outcome selection','evaluation':'Historical96+32 evaluation; newly frozen32lexical+32joint examples; test nouns disjoint from training and inspected historical authority pools','confounds':'Data size and lexical diversity increase together. New test nouns are related occupational words selected in fixed candidate order. Source-position context only. No manifold or multi-task transfer established.'}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');assert not (out/'captures.npz').exists()
_,decoder,_,_,_=runner.load_bound();torch.set_num_threads(6)
model,receipt=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
F=torch.nn.functional;axis=torch.tensor(decoder['frozen_decoder']['axis'],device='cuda',dtype=torch.float64);threshold=float(decoder['frozen_decoder']['threshold']);old_capture=runner.component._capture;captures=[];pos=None

def capture(*args):
 ret=old_capture(*args);raw,_,_,pp,_=ret;ix=torch.arange(len(raw),device=raw.device);captures.append((torch.stack([p[ix,pos] for p in pp],1).cpu().numpy(),raw[ix,pos].cpu().double().numpy()));return ret
runner.component._capture=capture
batches=[];max_replay=0
for st in range(0,len(rows),32):
 rr=rows[st:st+32];ids=torch.tensor([r['token_ids'] for r in rr],device='cuda');pos=torch.tensor([r['position'] for r in rr],device='cuda');answers=torch.tensor([[r['answer'],r['foil']] for r in rr],device='cuda');captures=[]
 result=runner.component.evaluate(model,ids,pos,answers,axis,threshold,torch,F)
 native=runner.component.native_margin(model,ids,pos,answers,torch,F);max_replay=max(max_replay,float((native-result['base_margin']).abs().max()))
 assert len(captures)==2
 batches.append(dict(clean=captures[0][0],delta=captures[1][0]-captures[0][0],raw=captures[0][1],Y=result['atoms'].cpu().numpy().T,target=result['target'].cpu().numpy(),native=native.cpu().numpy()))
 print('captured',st+len(rr),flush=True)
d={k:np.concatenate([b[k] for b in batches]) for k in batches[0]};d['X']=np.concatenate([d['clean'].reshape(len(rows),-1),d['delta'].reshape(len(rows),-1),d['raw']],1)
np.savez(out/'captures.npz',**d)
old=np.load(A/'causal_source_prediction/captures.npz');oldmeta=json.loads((A/'causal_source_prediction/metadata.json').read_text());meta=oldmeta+[{'split':r['split'],'token_sha256':hashlib.sha256(np.asarray(r['token_ids'],dtype=np.int64).tobytes()).hexdigest()} for r in rows]
assert len({m['token_sha256'] for m in meta})==len(meta)
allx={k:np.concatenate([old[k],d[k]]) for k in ['X','Y','target','clean','delta','raw']};X=allx['X'];Y=allx['Y'];target=allx['target']
train=np.array([i for i,m in enumerate(meta) if m['split'] in ['discovery','expanded_discovery']]);assert len(train)==224
xm=X[train].mean(0);_,_,vh=np.linalg.svd(X[train]-xm,full_matrices=False);ym=Y[train].mean(0)

def build_features(x,clean,delta,center,basis,train_z_scale=None):
 norms=np.concatenate([np.linalg.norm(clean,axis=2),np.linalg.norm(delta,axis=2),np.sum(clean*delta,axis=2)/(np.linalg.norm(clean,axis=2)*np.linalg.norm(delta,axis=2)).clip(1e-12)],1)
 f={'norms15':norms};scales={}
 for rank in [4,8]:
  z=(x-center)@basis[:rank].T;f[f'pca{rank}_linear']=z
  scale=z[train].std(0).clip(1e-12) if train_z_scale is None else train_z_scale[rank];scales[rank]=scale;z=z/scale;ii,jj=np.triu_indices(rank);f[f'pca{rank}_quadratic']=np.concatenate([z,z[:,ii]*z[:,jj]],1)
 return f,scales
features,scales=build_features(X,allx['clean'],allx['delta'],xm,vh)
predictions={'expanded_mean':np.broadcast_to(ym,Y.shape).copy()};saved={'train':train,'pca_mean':xm,'pca_basis':vh[:8],'output_mean':ym};normal=[]
for name,f in features.items():
 fm=f[train].mean(0);fs=f[train].std(0).clip(1e-12);h=(f-fm)/fs;ht=h[train];gram=ht.T@ht/len(train)+.1*np.eye(h.shape[1]);rhs=ht.T@(Y[train]-ym)/len(train);w=np.linalg.solve(gram,rhs);predictions['expanded_'+name]=ym+h@w;normal.append(float(np.linalg.norm(gram@w-rhs)/np.linalg.norm(rhs)))
 saved.update({name+'_features':f,name+'_mean':fm,name+'_scale':fs,name+'_weights':w})
# Evaluate frozen H1011 predictors on exactly the same new states.
oldscales={rank:(((old['X'][old['train']]-old['pca_mean'])@old['pca_basis'][:rank].T).std(0).clip(1e-12)) for rank in [4,8]}
ff,_=build_features(X,allx['clean'],allx['delta'],old['pca_mean'],old['pca_basis'],oldscales)
predictions['old_mean']=np.broadcast_to(old['output_mean'],Y.shape).copy()
for name,f in ff.items():predictions['old_'+name]=old['output_mean']+(f-old[name+'_mean'])/old[name+'_scale']@old[name+'_weights']
summary=[]
for split in sorted({m['split'] for m in meta}):
 ix=[i for i,m in enumerate(meta) if m['split']==split]
 for name,p in predictions.items():
  summary.append(dict(split=split,model=name,n=len(ix),joint_relative_l2=float(np.linalg.norm(p[ix].sum(1)-target[ix])/np.linalg.norm(target[ix])),edge_relative_l2=(np.linalg.norm(p[ix]-Y[ix],axis=0)/np.linalg.norm(Y[ix],axis=0)).tolist()))
controls={}
for split in sorted({r['split'] for r in rows}):
 ix=[i for i,r in enumerate(rows) if r['split']==split];controls[split]={'native_accuracy':float(np.mean(d['native'][ix]>0)),'teacher_joint_relative_l2':float(np.linalg.norm(d['Y'][ix].sum(1)-d['target'][ix])/np.linalg.norm(d['target'][ix]))}
for name,p in predictions.items():saved['prediction_'+name]=p
saved.update(allx);np.savez(out/'models_and_predictions.npz',**saved);(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
result={'config':config,'summary':summary,'controls':controls,'audits':{'native_replay_max_abs':max_replay,'panel_hash_unchanged':sha(panel_path)==config['panel_sha256'],'train_rows':len(train),'unique_token_sequences':len(meta),'normal_equation_relative_residual':max(normal),'finite':all(np.isfinite(p).all() for p in predictions.values()),'student_suffix_inputs':0,'new_reserved_test_examples':0,'new_native_sequences':len(rows),'new_partial_sequences':2*len(rows),'new_suffix_sequences':11*len(rows)}}
assert max_replay<1e-5 and result['audits']['finite']
(A/f'{S}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(controls),flush=True)
