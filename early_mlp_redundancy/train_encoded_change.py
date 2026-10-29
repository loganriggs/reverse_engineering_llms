"""M69 six prospectively fixed encoded-change fits; retain all final states."""
from pathlib import Path
import json,torch
from torch.nn import functional as F
from encoded_change_loss import EncodedExperiment
from balanced_response_loss import P,V,A,sha
from balanced_training_runtime import load_natural,natural_loss

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');pp=V/'encoded_change_training_protocol.json';q=json.loads(pp.read_text());dependencies={'balanced_response_training_protocol.json':'reference_training_protocol_sha256','encoded_change_objective_preflight_protocol.json':'preflight_protocol_sha256','encoded_change_calibration.json':'calibration_sha256','encoded_change_calibration_audit.json':'calibration_audit_sha256','encoded_change_gradient_checks.json':'endpoint_gradient_sha256','encoded_change_parameter_checks.json':'parameter_gradient_sha256','balanced_response_training_results.json':'controls_sha256'}
 for name,key in dependencies.items():assert sha(V/name)==q[key]
 for name in ['encoded_change_calibration_audit.json','encoded_change_gradient_checks.json','encoded_change_parameter_checks.json']:assert json.loads((V/name).read_text())['status']=='passed'
 assert sha(P/'encoded_change_loss.py')==q['loss_source_sha256'] and sha(P/'balanced_training_runtime.py')==q['natural_runtime_sha256'];ep=V/'encoded_change_evaluation_protocol.json';assert json.loads(ep.read_text())['training_protocol_sha256']==sha(pp);e=EncodedExperiment();natural,np=load_natural(e);cal=json.loads((V/'encoded_change_calibration.json').read_text());oldreceipt=json.loads((V/'balanced_response_training_receipt.json').read_text());assert sha(P/'balanced_response_loss.py')==oldreceipt['task_loss_sha256'] and sha(P/'balanced_training_runtime.py')==oldreceipt['runtime_sha256'];smoke=V/'cached_load_smoke.json';assert str(torch.__version__)==json.loads(smoke.read_text())['torch_version'];controls=json.loads((V/'balanced_response_training_results.json').read_text())['models']
 receipt=dict(protocol_sha256=sha(pp),evaluation_sha256=sha(ep),dependencies={name:sha(V/name) for name in dependencies},source_sha256=sha(Path(__file__)),loss_source_sha256=sha(P/'encoded_change_loss.py'),runtime_sha256=sha(P/'balanced_training_runtime.py'),old_task_loss_sha256=sha(P/'balanced_response_loss.py'),cache_sha256=sha(e.cachepath),checkpoint_sha256=e.receipt.weights_sha256,torch_version=str(torch.__version__),cuda_version=torch.version.cuda,version_reference_sha256=sha(smoke),version_scope='Current torch version matches historical cached-load smoke. M61 receipt itself did not store a version; M68 replayed all792 initial taskK values and unchanged natural runtime. Optimizer defaults checked against original saved optimizer states below. No full historical trajectory replay.')
 rp=V/'encoded_change_training_receipt.json'
 if rp.exists():assert json.loads(rp.read_text())==receipt
 else:rp.write_text(json.dumps(receipt,indent=2)+'\n')
 op=V/'encoded_change_training_results.json';assert not op.exists();dest=A/'emlp_m69_encoded_training';dest.mkdir(exist_ok=True);entries=[];optimizer_checks=[]
 for initial in e.protocol['models']:
  kind,seed=initial['kind'],initial['seed'];arm=q['arm'];schedule=e.protocol['schedules'][str(seed)];path=dest/f'{kind}_{seed}_{arm}.pt';progress=path.with_suffix('.progress.pt')
  if path.exists():
   s=torch.load(path,weights_only=True);assert s['receipt']==receipt and s['initial_sha256']==initial['sha256'] and len(s['history'])==132;entries.append(dict(kind=kind,seed=seed,arm=arm,path=str(path),sha256=sha(path)));optimizer_checks.append(s['optimizer_default_check']);print('REUSE',kind,seed,flush=True);continue
  f,mu,scale=e.featurizer(initial);opt=torch.optim.Adam(f.parameters(),lr=e.protocol['learning_rate']);history=[];start=0;control=next(x for x in controls if (x['kind'],x['seed'],x['arm'])==(kind,seed,'intact_KL'));oldprogress=Path(control['path']).with_suffix('.progress.pt');old=torch.load(oldprogress,weights_only=True);savedgroup={k:v for k,v in old['optimizer']['param_groups'][0].items() if k!='params'};newgroup={k:v for k,v in opt.state_dict()['param_groups'][0].items() if k!='params'};assert savedgroup==newgroup;optcheck=dict(kind=kind,seed=seed,control_progress_sha256=sha(oldprogress),defaults_equal=True);optimizer_checks.append(optcheck);del old
  if progress.exists():
   s=torch.load(progress,weights_only=True);assert s['receipt']==receipt and s['initial_sha256']==initial['sha256'];f.load_state_dict(s['state']);opt.load_state_dict(s['optimizer']);history=s['history'];start=len(history);assert 0<=start<=132
  for step in range(start,132):
   opt.zero_grad();parts=[]
   if step%2==0:
    doc=schedule['natural_documents'][step//2];c=natural[doc];pred,support,_=f((c['write'].cuda()-mu)/scale);K,_=natural_loss(e,pred*scale+mu,c);K.backward();parts.append(dict(key=f'natural:{doc}',K=float(K.detach()),loss=float(K.detach()),active_mean=float(support.sum(-1).mean())))
   else:
    assignment=schedule['task_updates'][step//2]
    for ep_ in [0,1]:
     key=assignment['pair_id']+f'|{ep_}';c=e.cache[key];d=e.cache[e.donor_key(key)];assert c['family']==assignment['family'];pr,support,_=f((c['write'].cuda()-mu)/scale);pd,ds,_=f((d['write'].cuda()-mu)/scale);K,R,B,_,_=e.encoded_losses(pr*scale+mu,pd*scale+mu,key);weight=cal['encoded_weight'];loss=(K+weight*B)/2;assert torch.isfinite(loss);loss.backward();parts.append(dict(key=key,donor_key=e.donor_key(key),family=c['family'],K=float(K.detach()),raw_R=float(R.detach()),balanced_R=float(B.detach()),weight=weight,loss=float(loss.detach()),active_mean=float(support.sum(-1).mean()),donor_active_mean=float(ds.sum(-1).mean())))
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters());assert all(p.grad is None for p in e.model.parameters());opt.step()
   with torch.no_grad():f.dictionary.copy_(F.normalize(f.dictionary,dim=0));assert torch.allclose(f.dictionary.norm(dim=0),torch.ones(128,device='cuda'),rtol=1e-5,atol=1e-6)
   history.append(dict(step=step,parts=parts,loss=sum(p['loss'] for p in parts)))
   if (step+1)%11==0 or step==131:
    tmp=progress.with_suffix('.tmp');torch.save(dict(receipt=receipt,initial_sha256=initial['sha256'],state=f.state_dict(),optimizer=opt.state_dict(),history=history),tmp);tmp.replace(progress)
   if (step+1)%33==0:print('UPDATES',kind,seed,step+1,flush=True)
  tmp=path.with_suffix('.tmp');torch.save(dict(receipt=receipt,initial_sha256=initial['sha256'],kind=kind,seed=seed,arm=arm,state={k:v.detach().cpu() for k,v in f.state_dict().items()},mean=mu.cpu(),scale=scale.cpu(),history=history,optimizer_default_check=optcheck),tmp);tmp.replace(path);entries.append(dict(kind=kind,seed=seed,arm=arm,path=str(path),sha256=sha(path)));print('FIT_COMPLETE',kind,seed,flush=True)
 op.write_text(json.dumps(dict(status='fits_completed_evaluation_pending',receipt=receipt,models=entries,optimizer_updates=792,forward_calls_this_process=e.calls,natural_cache_path=str(np),natural_cache_sha256=sha(np),optimizer_default_checks=optimizer_checks,scope='All six fixed final states retained. Online training losses are not evaluated fidelity or transfer. No candidate selection.'),indent=2)+'\n');print('DONE',len(entries),e.calls,flush=True)
if __name__=='__main__':main()
