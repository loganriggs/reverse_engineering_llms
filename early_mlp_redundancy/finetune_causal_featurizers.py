"""M8 fixed objective comparison: MSE versus native downstream KL."""
import json,hashlib,time
from pathlib import Path
import torch
from torch.nn import functional as F
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from probe_token_table_causal import score
P=Path(__file__).resolve().parent;A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');gp=P/'validation/causal_gradient_check.json';assert json.loads(gp.read_text())['status']=='passed';m7p=P/'validation/write_featurizers.json';m7=json.loads(m7p.read_text());cp=P/'validation/natural_token_context_cohort.json';cohort=json.loads(cp.read_text());m4=json.loads((P/'validation/natural_token_context.json').read_text());cache=Path(m4['cache']);assert sha(cache)==m4['cache_sha256'];data=torch.load(cache,weights_only=True);Y=data['Y'][1];starts=[x for x in m7['models'] if x['kind'] in ['scalar','block']]
 for x in starts:assert sha(Path(x['path']))==x['sha256']
 streams={}
 for seed in [71,72]:
  g=torch.Generator().manual_seed(8000+seed);streams[seed]=torch.cat([torch.randperm(32,generator=g) for _ in range(4)]).tolist()
 protocol=dict(source_sha256=sha(Path(__file__)),M7_sha256=sha(m7p),cohort_sha256=sha(cp),gradient_check_sha256=sha(gp),initial_models=starts,objectives=['reconstruction','downstream_KL'],steps=128,learning_rate=.0003,optimizer='Fresh Adam for each arm; same lr/steps, no optimizer-state transfer or gradient clipping.',document_streams=streams,design='Four saved scalar/block models×two objectives. Same initial weights, training normalization, active32 and perseed full-document stream. Each step128positions of one training document0–31, four permutations. MSE normalized rawwrite loss OR pure native-output KL overall128positions and50304logits. No mixed objective or coefficient search. Native model frozen; selected-support gradients through unwrapped validated native trace. Unit decoder columns normalized after eachupdate.',evaluation='Fixed finalstep, reused development documents32–63; allpositionwrite replacement, Downbiasretained. Raw/token/context errors, activecounts andoutputKL/CE/top1. No eval-based stopping, no heldouttask/BSFreserveddata or newcorpus.',limitations='Matchedupdates/data andinitialweights, notmatchedFLOPs. KLarmrequiresnative downstream forwards/backwards andteacher outputs. Only32trainingdocuments,128updates; nopopulation/convergence/manifold/newtaskclaim.')
 pp=P/'validation/causal_featurizer_protocol.json';pp.write_text(json.dumps(protocol,indent=2)+'\n');model,receipt=load_model();model.requires_grad_(False);device=next(model.parameters()).device;assert receipt.weights_sha256==m4['checkpoint']['weights_sha256'];teacher={};forward_calls=0
 for d in range(64):
  ids=torch.tensor([cohort['token_ids'][d]],device=device);out,cap=trace(model,ids);forward_calls+=1;assert torch.equal(model.transformer.h[1].mlp.Down(cap[1]['products'])[0].cpu(),Y[d*128:(d+1)*128]);teacher[d]=out.cpu()
 print('TEACHERS64',flush=True);del out,cap;dest=A/'emlp_m8_causal_featurizers';dest.mkdir(exist_ok=True);models=[];training=[];records=[];samples=[]
 for initial in starts:
  state=torch.load(initial['path'],weights_only=True);kind=initial['kind'];seed=initial['seed'];mu=state['mean'].to(device);scale=state['scale'].to(device);X=(Y.to(device)-mu)/scale
  for objective in ['reconstruction','downstream_KL']:
   m=Featurizer(kind,seed).to(device);m.load_state_dict(state['state']);opt=torch.optim.Adam(m.parameters(),lr=.0003);history=[];begin=time.time()
   for step,d in enumerate(streams[seed]):
    x=X[d*128:(d+1)*128];pred,_,_=m(x)
    if objective=='reconstruction':loss=F.mse_loss(pred,x)
    else:
     ids=torch.tensor([cohort['token_ids'][d]],device=device);q=(pred*scale+mu)[None];lp=teacher[d].to(device).double().log_softmax(-1)
     out,_cap=trace.__wrapped__(model,ids,replace=lambda i,z,w:q+model.transformer.h[1].mlp.Down_bias if i==1 else w);forward_calls+=1;loss=(lp.exp()*(lp-out.double().log_softmax(-1))).sum(-1).mean()
    assert torch.isfinite(loss);opt.zero_grad();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());opt.step()
    with torch.no_grad():m.dictionary.copy_(F.normalize(m.dictionary,dim=0))
    if step%32==0 or step==127:history.append(dict(step=step+1,document=d,loss=float(loss.detach())))
    if step%32==31:print('UPDATES',kind,seed,objective,step+1,flush=True)
   assert all(p.grad is None for p in model.parameters())
   with torch.no_grad():
    pred,q,code=m(X[4096:]);recon=pred*scale+mu;z=Y[4096:].double().to(device);error=recon.double()-z;token=data['tokens'][4096:].to(device);between=torch.empty_like(error);target_between=torch.empty_like(z)
    for t in token.unique():sel=token==t;between[sel]=error[sel].mean(0);target_between[sel]=z[sel].mean(0)
    within=error-between;tw=z-target_between;active=q.sum(-1);assert torch.all(active==32)
    rec=dict(kind=kind,seed=seed,objective=objective,history=history,eval_relative_raw_error=float(error.square().sum()/z.square().sum()),eval_token_mean_error_energy=float(between.square().sum()),eval_within_token_error_energy=float(within.square().sum()),eval_within_token_error_relative_to_within_target=float(within.square().sum()/tw.square().sum()),active_mean=float(active.mean()),active_min=int(active.min()),active_max=int(active.max()),elapsed_seconds=time.time()-begin)
    path=dest/f'{kind}_{seed}_{objective}.pt';torch.save(dict(state={k:v.cpu() for k,v in m.state_dict().items()},mean=mu.cpu(),scale=scale.cpu(),kind=kind,seed=seed,objective=objective,reconstructions=recon.cpu(),record=rec),path);models.append(dict(kind=kind,seed=seed,objective=objective,path=str(path),sha256=sha(path)));training.append(rec)
    for d in range(32,64):
     ids=torch.tensor([cohort['token_ids'][d]],device=device);ref=teacher[d].to(device);write=recon[(d-32)*128:(d-31)*128][None];out,_=trace(model,ids,replace=lambda i,z,w:write+model.transformer.h[1].mlp.Down_bias if i==1 else w);forward_calls+=1;met=score(ref,out,ids);base=score(ref,ref,ids);records.append(dict(document=d,kind=kind,seed=seed,objective=objective,native_CE=base['CE'],delta_CE=met['CE']-base['CE'],**met))
     if d==32:samples.append(dict(document=d,kind=kind,seed=seed,objective=objective,ids=ids.cpu(),native=ref.cpu(),output=out.cpu()))
   print('COMPLETED',kind,seed,objective,flush=True);del m,opt
 summaries=[]
 for m in models:
  rs=[x for x in records if all(x[k]==m[k] for k in ['kind','seed','objective'])];summaries.append({k:m[k] for k in ['kind','seed','objective']}|{k:sum(x[k] for x in rs)/32 for k in ['KL','CE','delta_CE','top1_agreement']})
 sp=A/'emlp_m8_causal_samples.pt';torch.save(samples,sp);out=dict(status='completed',protocol_sha256=sha(pp),models=models,training=training,causal_records=records,causal_summaries=summaries,forward_count=forward_calls,samples_path=str(sp),samples_sha256=sha(sp));(P/'validation/causal_featurizers.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()
