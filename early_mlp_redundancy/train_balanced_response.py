"""M61 fixed18 fits; retain final checkpoints, resume only exact receipts."""
from pathlib import Path
import json,torch
from torch.nn import functional as F
from balanced_response_loss import Experiment,P,V,A,sha
from balanced_training_runtime import load_natural,natural_loss

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');e=Experiment();natural,np=load_natural(e);cp=V/'balanced_response_calibration.json';ap=V/'balanced_response_calibration_audit.json';audit=json.loads(ap.read_text());assert audit['status']=='passed' and audit['protocol_sha256']==sha(e.pp) and audit['calibration_sha256']==sha(cp);assert audit['gradient_sha256']==sha(V/'balanced_response_gradient_checks.json');g=json.loads((V/'balanced_response_gradient_checks.json').read_text());assert g['loss_source_sha256']==sha(P/'balanced_response_loss.py');assert g['cache_sha256']==sha(e.cachepath);checkp=V/'balanced_natural_preflight.json';check=json.loads(checkp.read_text());assert check['status']=='passed' and check['runtime_sha256']==sha(P/'balanced_training_runtime.py') and check['runner_sha256']==sha(Path(__file__)) and check['protocol_sha256']==sha(e.pp)
 op=V/'balanced_response_training_results.json';assert not op.exists();dest=A/'emlp_m61_balanced_training';dest.mkdir(exist_ok=True);receipt=dict(protocol_sha256=sha(e.pp),calibration_sha256=sha(cp),calibration_audit_sha256=sha(ap),task_gradient_sha256=sha(V/'balanced_response_gradient_checks.json'),natural_preflight_sha256=sha(checkp),source_sha256=sha(Path(__file__)),task_loss_sha256=sha(P/'balanced_response_loss.py'),runtime_sha256=sha(P/'balanced_training_runtime.py'),checkpoint_sha256=e.receipt.weights_sha256);rp=V/'balanced_response_training_receipt.json'
 if rp.exists():assert json.loads(rp.read_text())==receipt
 else:rp.write_text(json.dumps(receipt,indent=2)+'\n')
 entries=[]
 for initial in e.protocol['models']:
  kind,seed=initial['kind'],initial['seed'];schedule=e.protocol['schedules'][str(seed)]
  for arm in e.protocol['arms']:
   path=dest/f'{kind}_{seed}_{arm}.pt';progress=path.with_suffix('.progress.pt')
   if path.exists():
    s=torch.load(path,weights_only=True);assert s['receipt']==receipt and s['initial_sha256']==initial['sha256'] and len(s['history'])==132;entries.append(dict(kind=kind,seed=seed,arm=arm,path=str(path),sha256=sha(path)));print('REUSE',kind,seed,arm,flush=True);continue
   f,mu,scale=e.featurizer(initial);opt=torch.optim.Adam(f.parameters(),lr=e.protocol['learning_rate']);history=[];start=0
   if progress.exists():
    s=torch.load(progress,weights_only=True);assert s['receipt']==receipt and s['initial_sha256']==initial['sha256'];f.load_state_dict(s['state']);opt.load_state_dict(s['optimizer']);history=s['history'];start=len(history);assert 0<=start<=132
   for step in range(start,e.protocol['steps']):
    opt.zero_grad();parts=[]
    if step%2==0:
     doc=schedule['natural_documents'][step//2];c=natural[doc];pred,support,_=f((c['write'].cuda()-mu)/scale);K,_=natural_loss(e,pred*scale+mu,c);K.backward();parts.append(dict(key=f'natural:{doc}',K=float(K.detach()),loss=float(K.detach()),active_mean=float(support.sum(-1).mean())))
    else:
     assignment=schedule['task_updates'][step//2]
     for ep in [0,1]:
      key=assignment['pair_id']+f'|{ep}';c=e.cache[key];assert c['family']==assignment['family'];pred,support,_=f((c['write'].cuda()-mu)/scale);K,R,B,_=e.losses(pred*scale+mu,c);weight=0 if arm=='intact_KL' else e.cal['raw_weight'] if arm=='KL_raw_response' else e.cal['balanced_weight'];response=R if arm=='KL_raw_response' else B;loss=(K+weight*response)/2;assert torch.isfinite(loss);loss.backward();parts.append(dict(key=key,family=c['family'],K=float(K.detach()),raw_R=float(R.detach()),balanced_R=float(B.detach()),weight=weight,loss=float(loss.detach()),active_mean=float(support.sum(-1).mean())))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters());assert all(p.grad is None for p in e.model.parameters());opt.step()
    with torch.no_grad():f.dictionary.copy_(F.normalize(f.dictionary,dim=0));assert torch.allclose(f.dictionary.norm(dim=0),torch.ones(128,device='cuda'),rtol=1e-5,atol=1e-6)
    history.append(dict(step=step,parts=parts,loss=sum(p['loss'] for p in parts)));assert len(history)==step+1
    if (step+1)%11==0 or step==131:
     tmp=progress.with_suffix('.tmp');torch.save(dict(receipt=receipt,initial_sha256=initial['sha256'],state=f.state_dict(),optimizer=opt.state_dict(),history=history),tmp);tmp.replace(progress)
    if (step+1)%33==0:print('UPDATES',kind,seed,arm,step+1,flush=True)
   tmp=path.with_suffix('.tmp');torch.save(dict(receipt=receipt,initial_sha256=initial['sha256'],kind=kind,seed=seed,arm=arm,state={k:v.detach().cpu() for k,v in f.state_dict().items()},mean=mu.cpu(),scale=scale.cpu(),history=history),tmp);tmp.replace(path);entries.append(dict(kind=kind,seed=seed,arm=arm,path=str(path),sha256=sha(path)));print('FIT_COMPLETE',kind,seed,arm,flush=True)
 op.write_text(json.dumps(dict(status='fits_completed_evaluation_pending',receipt=receipt,models=entries,optimizer_updates=18*132,forward_calls_this_process=e.calls,natural_cache_path=str(np),natural_cache_sha256=sha(np),scope='Fixed final states; online losses are not final retention or transfer. No candidate selection.'),indent=2)+'\n');print('DONE',len(entries),flush=True)
if __name__=='__main__':main()
