"""H1009: predict early MLP writes from inputs, document-separated pilot."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from pathlib import Path
import torch,torch.nn.functional as F,json,time
S='bilin18_early_mlp_prediction'
def features(z,degree):
 if degree==1:return z
 idx=torch.triu_indices(z.shape[1],z.shape[1]);return torch.cat([z,z[:,idx[0]]*z[:,idx[1]]],1)
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'natural':ART/'bilin18_natural_residual_discovery_captures.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['natural'],weights_only=False);tokens=data['tokens'];meta=data['metadata'];positions=[31,63,95,119];assert len(meta)==1024
 train=[i for i,r in enumerate(meta) if r['split']=='discovery'];validation=[i for i,r in enumerate(meta) if r['split']=='confirmation'];assert len(train)==768 and len(validation)==256;assert not {meta[i]['document_hash'] for i in train}&{meta[i]['document_hash'] for i in validation}
 cfg=dict(experiment='H1009',hashes=hashes,design='Capture normalized inputs and MLP writes in layers0,1,2 at fourfixedpositions on256historical documents. Fit each local predictor on768states/192discovery documents; evaluate256states/64confirmation documents. No output activation supplied to predictor.',models='Mean baseline; rank16/rank32 input PCA then linear or full degree2 polynomial ridge prediction of1152dimensional write. PCA, feature centering/scaling and target mean fit on discovery only. Fixed ridge penalty0.01 in mean squared standardized-feature objective; no validation model selection.',metrics='1-sum squared prediction error/sum squared deviation from training target mean; negative scores allowed. Also output-PCA oracle rank32 for compression-only comparison, explicitly given true output.',limits='Teacher supplies native local input; not replacement of full attention+MLP block or first-three-layer prefix. Quadratic student is an empirical polynomial surrogate, not a semantic explanation. Existing development documents reused. Fourpositions perdocument not independent samples.',holdouts='Only H998discovery/confirmation subset of original historical BSFtraining documents; original BSFvalidation/test not used.')
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 inputs={l:[] for l in range(3)};outputs={l:[] for l in range(3)}
 def prefix(ids,pos):
  xx={};yy={};handles=[]
  for l in range(3):
   def hook(m,a,out,l=l):xx[l]=a[0][:,pos].detach().cpu();yy[l]=out[:,pos].detach().cpu()
   handles.append(model.blocks[l].mlp.register_forward_hook(hook))
  try:
   x0=F.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None
   for block in model.blocks[:3]:x,bus=block(x,bus,x0)
  finally:
   for h in handles:h.remove()
  return xx,yy
 for st in range(0,256,8):
  xx,yy=prefix(tokens[st:st+8].cuda(),positions)
  for l in range(3):inputs[l].append(xx[l]);outputs[l].append(yy[l])
 inputs={l:torch.cat(v).reshape(-1,1152).double() for l,v in inputs.items()};outputs={l:torch.cat(v).reshape(-1,1152).double() for l,v in outputs.items()};xx,yy=prefix(tokens[:1,:32].cuda(),[31]);replay=max(float((yy[l].reshape(-1)-outputs[l][0]).norm()/outputs[l][0].norm()) for l in range(3));assert replay<1e-5
 models={};predictions={};summary=[];normal_errors=[]
 for layer in range(3):
  X=inputs[layer];Y=outputs[layer];xm=X[train].mean(0);ym=Y[train].mean(0);_,sv,vh=torch.linalg.svd(X[train]-xm,full_matrices=False);models[layer]={};predictions[layer]={'mean':ym.expand_as(Y).clone()}
  _,sy,vy=torch.linalg.svd(Y[train]-ym,full_matrices=False);predictions[layer]['output_oracle32']=ym+(Y-ym)@vy[:32].T@vy[:32]
  for rank in [16,32]:
   basis=vh[:rank].T;z=(X-xm)@basis;zs=z[train].square().mean(0).sqrt().clamp_min(1e-8);z=z/zs
   for degree in [1,2]:
    f=features(z,degree);fm=f[train].mean(0);fs=(f[train]-fm).square().mean(0).sqrt().clamp_min(1e-8);h=(f-fm)/fs;ht=h[train];target=Y[train]-ym;gram=ht.T@ht/len(train)+.01*torch.eye(h.shape[1]);rhs=ht.T@target/len(train);weights=torch.linalg.solve(gram,rhs);err=float((gram@weights-rhs).norm()/rhs.norm());normal_errors.append(err);name=f'input{rank}_degree{degree}';pred=ym+h@weights;predictions[layer][name]=pred;models[layer][name]=dict(input_mean=xm,basis=basis,coordinate_scale=zs,feature_mean=fm,feature_scale=fs,output_mean=ym,weights=weights,degree=degree,feature_dimension=h.shape[1],parameters=weights.numel()+basis.numel())
  for split,ii in [('discovery',train),('confirmation',validation)]:
   den=(Y[ii]-ym).square().sum()
   for name,pred in predictions[layer].items():summary.append(dict(layer=layer,split=split,model=name,n=len(ii),relative_error=float((Y[ii]-pred[ii]).square().sum()/den),score=1-float((Y[ii]-pred[ii]).square().sum()/den)))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),three_block_prefix_evaluations=257,whole_model_forwards=0,causal_prefix_replay_relative=replay,ridge_normal_equation_relative_error=max(normal_errors),document_split_overlap=0,finite=all(torch.isfinite(v).all().item() for preds in predictions.values() for v in preds.values()),original_reserved_test_evaluations=0)
 assert audits['hashes_unchanged'] and audits['finite'] and max(normal_errors)<1e-8
 torch.save(dict(inputs=inputs,outputs=outputs,models=models,predictions=predictions,metadata=meta,train_indices=train,validation_indices=validation),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,summary=summary,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(summary,audits,flush=True)
if __name__=='__main__':main()
