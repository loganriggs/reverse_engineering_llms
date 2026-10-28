"""M7 fixed native MLP1-write reconstruction pilot and causal evaluation."""
import json,hashlib,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from inspect_model import load_model,trace
from probe_token_table_causal import score
P=Path(__file__).resolve().parent;A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');SEEDS=[71,72];KINDS=['scalar','block','shared_fixed']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Featurizer(nn.Module):
 def __init__(self,kind,seed):
  super().__init__();self.kind=kind;g=torch.Generator().manual_seed(seed);D=F.normalize(torch.randn(1152,128,generator=g),dim=0);self.dictionary=nn.Parameter(D);self.encoder=nn.Parameter(D.T.clone());self.encoder_bias=nn.Parameter(torch.zeros(128));self.bias=nn.Parameter(torch.zeros(1152));M=torch.zeros(32,128)
  if kind=='block':M[torch.arange(32)[:,None],torch.arange(128).reshape(32,4)]=1
  elif kind=='shared_fixed':
   for half in range(2):M[torch.arange(16)[:,None]+16*half,torch.randperm(128,generator=g).reshape(16,8)]=1
   assert torch.equal(M.sum(0),torch.full((128,),2.))
  self.register_buffer('membership',M)
 def forward(self,x):
  a=F.linear(x,self.encoder,self.encoder_bias)
  if self.kind=='scalar':q=torch.zeros_like(a).scatter_(-1,a.abs().topk(32,dim=-1).indices,1)
  else:
   energy=a.square()@self.membership.T;gates=torch.zeros_like(energy).scatter_(-1,energy.topk(8 if self.kind=='block' else 4,dim=-1).indices,1);q=((gates@self.membership)>0).to(a.dtype)
  code=a*q;return code@F.normalize(self.dictionary,dim=0).T+self.bias,q,code

def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');torch.manual_seed(71)
 cp=P/'validation/natural_token_context_cohort.json';mp=P/'validation/natural_token_context.json';c=json.loads(cp.read_text());m4=json.loads(mp.read_text());cache=Path(m4['cache']);assert sha(cache)==m4['cache_sha256'];assert sha(cp)==m4['cohort_sha256'];data=torch.load(cache,weights_only=True);Y=data['Y'][1];train=data['unique_indices'][data['documents'][data['unique_indices']]<32];test=torch.arange(4096,8192);assert not set(r['document_hash'] for r in c['records'][:32])&set(r['document_hash'] for r in c['records'][32:]);mu=Y[train].mean(0);scale=(Y[train]-mu).square().mean().sqrt();X=((Y-mu)/scale).cuda();train=train.cuda()
 protocol=dict(source_sha256=sha(Path(__file__)),cohort_sha256=sha(cp),M4_sha256=sha(mp),site='native MLP1 product write excluding Down bias',train_documents=list(range(32)),evaluation_documents=list(range(32,64)),train_unique_positions=len(train),eval_positions=len(test),seeds=SEEDS,kinds=KINDS,atoms=128,steps=1000,batch=128,learning_rate=.001,objective='Mean squared error on training-centered globally RMS-scaled RAW native writes. No task labels or downstream loss supervision.',selection='Fixed final step; no evaluation-based selection or tuning.',architecture='Signed linear encoder, unit-column decoder, learned encoder/decoder biases. ScalarTop32;32disjoint blocks of4 withTop8L2 gates;32fixed shared groups of8 withTop4L2 gates, balanced atom degree2. Exact union; decode shared coordinates once. Common trainable parameter count and initialization. Shared union <=32, so actual active costs reported; not perfectly matched active dimension. No learned gate encoder/membership, hierarchy, Grassmannian constraint or computational DAG.',scope='Tiny reused corpus development pilot. Existing first32documents fit, latter32evaluate; no new reserved BSF/test/task exposure. Models reconstruct already-computed writes, not executable replacements for computing MLP inputs. One training direction only. Fixed minibatch streams shared across kinds perseed. Token/context error decomposition within evaluation is descriptive.')
 pp=P/'validation/write_featurizer_protocol.json';pp.write_text(json.dumps(protocol,indent=2)+'\n');dest=A/'emlp_m7_write_featurizers';dest.mkdir(exist_ok=True);models=[];train_records=[]
 for seed in SEEDS:
  for kind in KINDS:
   m=Featurizer(kind,seed).cuda();opt=torch.optim.Adam(m.parameters(),lr=.001);gen=torch.Generator().manual_seed(seed+1000);history=[];start=time.time()
   for step in range(1000):
    idx=train[torch.randint(len(train),(128,),generator=gen).cuda()];pred,_,_=m(X[idx]);loss=F.mse_loss(pred,X[idx]);opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():m.dictionary.copy_(F.normalize(m.dictionary,dim=0))
    if step%200==0 or step==999:history.append(dict(step=step+1,loss=float(loss.detach())))
   with torch.no_grad():pred,q,code=m(X[test.cuda()]);recon=pred*scale.cuda()+mu.cuda();z=Y[test].double().cuda();err=recon.double()-z;token=data['tokens'][test].cuda();between=torch.empty_like(err);target_between=torch.empty_like(z)
   with torch.no_grad():
    for t in token.unique():
     select=token==t;between[select]=err[select].mean(0);target_between[select]=z[select].mean(0)
    within=err-between;tw=z-target_between;active=q.sum(-1);rec=dict(kind=kind,seed=seed,trainable_parameters=sum(p.numel() for p in m.parameters()),history=history,eval_relative_raw_error=float(err.square().sum()/z.square().sum()),eval_relative_centered_error=float(err.square().sum()/(z-z.mean(0)).square().sum()),eval_token_mean_error_energy=float(between.square().sum()),eval_within_token_error_energy=float(within.square().sum()),eval_within_token_error_relative_to_within_target=float(within.square().sum()/tw.square().sum()),active_mean=float(active.mean()),active_min=int(active.min()),active_max=int(active.max()),unused_atoms=int((code.abs().sum(0)==0).sum()),elapsed_training_seconds=time.time()-start)
    assert torch.allclose(err.square().sum(),between.square().sum()+within.square().sum(),rtol=1e-10);assert int(active.max())<=32;assert torch.isfinite(recon).all()
   state=dict(state={k:v.cpu() for k,v in m.state_dict().items()},mean=mu,scale=scale,kind=kind,seed=seed,reconstructions=recon.cpu(),record=rec);path=dest/f'{kind}_{seed}.pt';torch.save(state,path);models.append(dict(kind=kind,seed=seed,path=str(path),sha256=sha(path)));train_records.append(rec);print('TRAINED',kind,seed,rec['eval_relative_raw_error'],rec['active_mean'],flush=True);del m,opt
 with torch.no_grad():
  model,receipt=load_model();assert receipt.weights_sha256==m4['checkpoint']['weights_sha256'];device=next(model.parameters()).device;states=[torch.load(x['path'],weights_only=True) for x in models];records=[];samples=[];calls=0
  for d in range(32,64):
   ids=torch.tensor([c['token_ids'][d]],device=device);native,cap=trace(model,ids);calls+=1;assert torch.equal(model.transformer.h[1].mlp.Down(cap[1]['products'])[0].cpu(),Y[d*128:(d+1)*128]);base=score(native,native,ids);sample=dict(document=d,ids=ids.cpu(),native=native.cpu(),conditions=[]) if d==32 else None
   for state in states:
    pred=state['reconstructions'][(d-32)*128:(d-31)*128].to(device)[None]
    def replace(i,z,write):return pred+model.transformer.h[1].mlp.Down_bias if i==1 else write
    out,_=trace(model,ids,replace=replace);calls+=1;met=score(native,out,ids);records.append(dict(document=d,kind=state['kind'],seed=state['seed'],native_CE=base['CE'],delta_CE=met['CE']-base['CE'],**met))
    if sample is not None:sample['conditions'].append(dict(kind=state['kind'],seed=state['seed'],output=out.cpu()))
   if sample is not None:samples.append(sample)
   if d%8==7:print('EVALUATED_DOCUMENTS',d-31,flush=True)
 summaries=[]
 for m in models:
  rs=[x for x in records if x['kind']==m['kind'] and x['seed']==m['seed']];summaries.append(dict(kind=m['kind'],seed=m['seed'],**{k:sum(x[k] for x in rs)/len(rs) for k in ['KL','CE','delta_CE','top1_agreement']}))
 sp=A/'emlp_m7_causal_samples.pt';torch.save(samples,sp);out=dict(status='completed',protocol_sha256=sha(pp),models=models,training=train_records,causal_records=records,causal_summaries=summaries,forward_count=calls,samples_path=str(sp),samples_sha256=sha(sp));(P/'validation/write_featurizers.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()
