"""M82 fixed eight natural-reconstruction warmups with exact resume receipts."""
import json,hashlib
import torch
from torch.nn import functional as F
from bilinear_input_runtime import InputWritePredictor
from fit_pca_encoder import sha,V,A
P=V.parent

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');pp=V/'bilinear_input_warmup_protocol.json';q=json.loads(pp.read_text());assert sha(V/'bilinear_input_preflight_completion.json')==q['preflight_sha256'];assert sha(V/'bilinear_input_initial_states.json')==q['initial_states_sha256'];manifest=json.loads((V/'bilinear_input_initial_states.json').read_text());ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];s=torch.load(ref['path'],weights_only=True);m4=json.loads((V/'natural_token_context.json').read_text());assert sha(m4['cache'])==m4['cache_sha256'];data=torch.load(m4['cache'],weights_only=True);ix=s['indices'];assert len(ix)==4092 and bool((data['documents'][ix]<32).all());z=s['z'][ix].cuda();w=data['Y'][1][ix].cuda()
 receipt=dict(protocol_sha256=sha(pp),evaluation_sha256=sha(V/'bilinear_input_prospective_evaluation.json'),source_sha256=sha(__file__),module_sha256=sha(P/'bilinear_input_featurizer.py'),runtime_sha256=sha(P/'bilinear_input_runtime.py'),torch_version=str(torch.__version__),cuda_version=torch.version.cuda,reference_sha256=sha(V/'native_input_reference.json'),write_cache_sha256=m4['cache_sha256']);rp=V/'bilinear_input_warmup_receipt.json'
 if rp.exists():assert json.loads(rp.read_text())==receipt
 else:rp.write_text(json.dumps(receipt,indent=2)+'\n')
 entries=[]
 for entry in manifest['models']:
  assert sha(entry['path'])==entry['sha256'];initial=torch.load(entry['path'],weights_only=True);model=InputWritePredictor(initial).cuda();opt=torch.optim.Adam(model.parameters(),lr=q['learning_rate']);gen=torch.Generator().manual_seed(entry['seed']+1000);schedule=torch.randint(len(ix),(1000,128),generator=gen);schedule_hash=hashlib.sha256(schedule.numpy().tobytes()).hexdigest();path=A/f"emlp_m82_warmup_{entry['kind']}_{entry['seed']}.pt";progress=path.with_suffix('.progress.pt');history=[];start=0
  if path.exists():
   saved=torch.load(path,weights_only=True);assert saved['receipt']==receipt and saved['initial_sha256']==entry['sha256'] and len(saved['history'])==1000;entries.append(dict(kind=entry['kind'],seed=entry['seed'],path=str(path),sha256=sha(path)));continue
  if progress.exists():
   saved=torch.load(progress,weights_only=True);assert saved['receipt']==receipt and saved['initial_sha256']==entry['sha256'] and saved['schedule_sha256']==schedule_hash;model.load_state_dict(saved['state']);opt.load_state_dict(saved['optimizer']);history=saved['history'];start=len(history)
  for step in range(start,1000):
   index=schedule[step].cuda();prediction,_,_=model(z[index]);loss=((prediction-w[index])/model.output_scale).square().mean();opt.zero_grad();loss.backward();assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());opt.step()
   with torch.no_grad():model.features.dictionary.copy_(F.normalize(model.features.dictionary,dim=0))
   history.append(float(loss.detach()))
   if (step+1)%100==0:
    payload=dict(receipt=receipt,initial_sha256=entry['sha256'],state=model.state_dict(),optimizer=opt.state_dict(),history=history,schedule_sha256=schedule_hash);tmp=progress.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(progress)
   if (step+1)%500==0:print(entry['kind'],entry['seed'],step+1,flush=True)
  with torch.no_grad():
   error=0.;active=0
   for offset in range(0,len(z),128):
    pred,support,_=model(z[offset:offset+128]);error+=float(((pred-w[offset:offset+128])/model.output_scale).double().square().sum());active+=float(support.sum())
  state=dict(receipt=receipt,initial_sha256=entry['sha256'],state={k:v.cpu() for k,v in model.state_dict().items()},kind=entry['kind'],seed=entry['seed'],history=history,schedule_sha256=schedule_hash,full_training_MSE=error/(len(z)*1152),active_mean=active/len(z),optimizer_updates=1000,optimizer_defaults={k:v for k,v in opt.param_groups[0].items() if k!='params'});torch.save(state,path);entries.append(dict(kind=entry['kind'],seed=entry['seed'],path=str(path),sha256=sha(path)));print('FINISHED',entry['kind'],entry['seed'],state['full_training_MSE'],flush=True)
 out=V/'bilinear_input_warmup_results.json';assert not out.exists();out.write_text(json.dumps(dict(status='fits_completed_audit_pending',receipt=receipt,models=entries,optimizer_updates=8000,scope='NaturaltrainingMSEwarmup only; no heldoutscore ormodelselection, downstreamcausaltraining notstarted.'),indent=2)+'\n')
if __name__=='__main__':main()
