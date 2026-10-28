"""M36 execute twelve fixed common-data/objective fits; evaluate separately."""
from pathlib import Path
import json
import torch
from torch.nn import functional as F
from joint_shared_loss import Experiment,P,A,sha
from inspect_model import trace

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');e=Experiment();cp=P/'joint_shared_calibration.json';cal=json.loads(cp.read_text());audit=json.loads((P/'joint_shared_calibration_audit.json').read_text());assert audit['status']=='passed' and audit['calibration_sha256']==sha(cp) and audit['protocol_sha256']==sha(e.pp);assert audit['gradient_sha256']==sha(P/'joint_shared_gradient_checks.json');assert cal['loss_source_sha256']==sha(P.parent/'joint_shared_loss.py');weight=cal['response_weight']
 op=P/'joint_shared_training_results.json';assert not op.exists(),'Training already completed; inspect results instead of duplicating.'
 dest=A/'emlp_m36_joint_shared_training';dest.mkdir(exist_ok=True);receipt=dict(protocol_sha256=sha(e.pp),calibration_sha256=sha(cp),audit_sha256=sha(P/'joint_shared_calibration_audit.json'),source_sha256=sha(Path(__file__)),loss_source_sha256=sha(P.parent/'joint_shared_loss.py'),checkpoint_sha256=e.receipt.weights_sha256)
 rp=P/'joint_shared_training_receipt.json'
 if rp.exists():assert json.loads(rp.read_text())==receipt
 else:rp.write_text(json.dumps(receipt,indent=2)+'\n')
 cohortpath=P/'natural_token_context_cohort.json';cohort=json.loads(cohortpath.read_text());nc=A/'emlp_m36_natural_training_cache.pt'
 if nc.exists():
  saved=torch.load(nc,weights_only=True);assert saved['cohort_sha256']==sha(cohortpath) and saved['checkpoint_sha256']==e.receipt.weights_sha256;natural=saved['natural']
 else:
  natural={}
  with torch.no_grad():
   for d in range(32):
    ids=torch.tensor([cohort['token_ids'][d]],device=e.device);out,cap=trace(e.model,ids);natural[d]=dict(ids=ids.cpu(),native=out.cpu(),write=e.mlp.Down(cap[1]['products']).cpu())
  tmp=nc.with_suffix('.tmp');torch.save(dict(cohort_sha256=sha(cohortpath),checkpoint_sha256=e.receipt.weights_sha256,natural=natural),tmp);tmp.replace(nc)
 print('NATURAL_CACHE_READY',len(natural),flush=True)
 assignments={(x['seed'],x['task_update'],x['endpoint']):x for x in e.p['target_assignments']};models=[]
 for initial in e.p['initial_models']:
  kind,seed=initial['kind'],initial['seed'];schedule=e.p['schedules'][str(seed)]
  for arm in e.p['arms']:
   path=dest/f'{kind}_{seed}_{arm}.pt'
   if path.exists():
    saved=torch.load(path,weights_only=True);assert saved['receipt']==receipt and saved['initial_sha256']==initial['sha256'];assert len(saved['history'])==120
    models.append(dict(kind=kind,seed=seed,arm=arm,path=str(path),sha256=sha(path)));print('REUSE_COMPLETE',kind,seed,arm,flush=True);continue
   f,mu,scale=e.featurizer(initial);opt=torch.optim.Adam(f.parameters(),lr=e.p['learning_rate']);history=[];start=0;progress=path.with_suffix('.progress.pt')
   if progress.exists():
    saved=torch.load(progress,weights_only=True);assert saved['receipt']==receipt and saved['initial_sha256']==initial['sha256'];f.load_state_dict(saved['state']);opt.load_state_dict(saved['optimizer']);history=saved['history'];start=len(history)
   for step in range(start,e.p['steps']):
    opt.zero_grad();parts=[]
    if step%2==0:batch=[(natural[schedule['natural_documents'][step]],[],f"natural:{schedule['natural_documents'][step]}")]
    else:
     batch=[]
     for endpoint in [0,1]:
      a=assignments[seed,step//2,endpoint];indices=a['target_indices'] if arm=='intact_KL_plus_native_response' else [];batch.append((e.cache['inputs'][a['input_key']],indices,a['input_key']))
    for c,indices,key in batch:
     pred,support,_=f((c['write'].to(e.device)-mu)/scale);q=pred*scale+mu;K,R,*_=e.losses(q,c,indices);loss=(K+weight*R)/len(batch);assert torch.isfinite(loss);loss.backward();parts.append(dict(key=key,target_indices=indices,K=float(K.detach()),R=float(R.detach()),scaled_loss=float(loss.detach()),active_mean=float(support.sum(-1).mean()),active_min=int(support.sum(-1).min()),active_max=int(support.sum(-1).max())))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters());opt.step()
    with torch.no_grad():f.dictionary.copy_(F.normalize(f.dictionary,dim=0))
    history.append(dict(step=step,parts=parts,loss=sum(x['scaled_loss'] for x in parts)))
    if (step+1)%30==0:
     tmp=progress.with_suffix('.tmp');torch.save(dict(receipt=receipt,initial_sha256=initial['sha256'],state=f.state_dict(),optimizer=opt.state_dict(),history=history),tmp);tmp.replace(progress);print('UPDATES',kind,seed,arm,step+1,flush=True)
   assert all(p.grad is None for p in e.model.parameters());state={k:v.detach().cpu() for k,v in f.state_dict().items()};tmp=path.with_suffix('.tmp');torch.save(dict(receipt=receipt,initial_sha256=initial['sha256'],kind=kind,seed=seed,arm=arm,state=state,mean=mu.cpu(),scale=scale.cpu(),history=history),tmp);tmp.replace(path);models.append(dict(kind=kind,seed=seed,arm=arm,path=str(path),sha256=sha(path)));print('FIT_COMPLETE',kind,seed,arm,flush=True)
 result=dict(status='fits_completed_evaluation_pending',receipt=receipt,models=models,optimizer_updates=12*120,forward_calls_this_process=e.calls,natural_cache_path=str(nc),natural_cache_sha256=sha(nc),scope='All fixed final states retained. Online training losses are not final retention or transfer measurements; evaluation pending.')
 assert len(models)==12;op.write_text(json.dumps(result,indent=2)+'\n');print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
