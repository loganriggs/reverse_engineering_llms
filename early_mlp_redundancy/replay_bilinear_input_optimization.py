"""M83 instrumentation-only exact replay of two fixed M82 trajectories."""
import hashlib,json
import torch
from torch.nn import functional as F
from bilinear_input_runtime import InputWritePredictor
from fit_pca_encoder import sha,V,A
P=V.parent

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');pp=V/'bilinear_input_optimization_replay_protocol.json';protocol=json.loads(pp.read_text())
 for name,digest in protocol['references'].items():assert sha(V/name)==digest
 receipt=json.loads((V/'bilinear_input_warmup_receipt.json').read_text());assert receipt['module_sha256']==sha(P/'bilinear_input_featurizer.py') and receipt['runtime_sha256']==sha(P/'bilinear_input_runtime.py');assert receipt['torch_version']==str(torch.__version__) and receipt['cuda_version']==torch.version.cuda
 ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];s=torch.load(ref['path'],weights_only=True);m4=json.loads((V/'natural_token_context.json').read_text());assert sha(m4['cache'])==m4['cache_sha256'];data=torch.load(m4['cache'],weights_only=True);ix=s['indices'];rare=int((ix==1280).nonzero().item());z=s['z'][ix].cuda();w=data['Y'][1][ix].cuda();initials=json.loads((V/'bilinear_input_initial_states.json').read_text())['models'];originals=json.loads((V/'bilinear_input_warmup_results.json').read_text())['models'];models=[]
 for kind in ['shared_fixed','dense']:
  entry=next(e for e in initials if e['kind']==kind and e['seed']==71);oldentry=next(e for e in originals if e['kind']==kind and e['seed']==71);assert sha(entry['path'])==entry['sha256'] and sha(oldentry['path'])==oldentry['sha256'];old=torch.load(oldentry['path'],weights_only=True);initial=torch.load(entry['path'],weights_only=True);f=InputWritePredictor(initial).cuda();opt=torch.optim.Adam(f.parameters(),lr=.001);g=torch.Generator().manual_seed(1071);schedule=torch.randint(4092,(1000,128),generator=g);assert hashlib.sha256(schedule.numpy().tobytes()).hexdigest()==old['schedule_sha256'];rows=[]
  @torch.no_grad()
  def snapshot():
   zz=z[rare:rare+1];xx=(zz-f.input_mean)/f.input_scale;left=F.linear(xx,f.features.left,f.features.left_bias);right=F.linear(xx,f.features.right,f.features.right_bias);pred,q,code=f(zz)
   return dict(error=float(((pred-w[rare:rare+1])/f.output_scale).double().square().mean()),left_norm=float(left.double().norm()),right_norm=float(right.double().norm()),code_norm=float(code.double().norm()),active_indices=q[0].nonzero().flatten().cpu().tolist(),code=code[0].cpu().tolist())
  before=snapshot()
  for step in range(1000):
   index=schedule[step].cuda();previous={n:p.detach().clone() for n,p in f.named_parameters()};prediction,_,_=f(z[index]);per=((prediction-w[index])/f.output_scale).square();loss=per.mean();opt.zero_grad();loss.backward();assert float(loss.detach())==old['history'][step],(kind,step,float(loss),old['history'][step]);gradient={n:float(p.grad.double().norm()) for n,p in f.named_parameters()};count=int((index==rare).sum());contribution=float(per[index==rare].detach().sum()/per.numel()) if count else 0.;opt.step()
   with torch.no_grad():f.features.dictionary.copy_(F.normalize(f.features.dictionary,dim=0))
   after=snapshot();update={n:float((p.detach()-previous[n]).double().norm()) for n,p in f.named_parameters()};rows.append(dict(step=step,rare_count=count,batch_loss=float(loss.detach()),rare_batch_loss_contribution=contribution,gradient_norms=gradient,update_norms=update,before=before,after=after,support_changed=before['active_indices']!=after['active_indices']));before=after
   if (step+1)%250==0:print(kind,step+1,flush=True)
  assert all(torch.equal(v.cpu(),old['state'][k]) for k,v in f.state_dict().items());oldprogress=torch.load(oldentry['path'].replace('.pt','.progress.pt'),weights_only=True)
  for k,state in opt.state_dict()['state'].items():
   for name,value in state.items():assert torch.equal(value.cpu(),oldprogress['optimizer']['state'][k][name].cpu())
  models.append(dict(kind=kind,seed=71,original_sha256=oldentry['sha256'],schedule_sha256=old['schedule_sha256'],history_exact=True,final_state_exact=True,optimizer_state_exact=True,rows=rows));print('EXACT',kind,flush=True)
 out=dict(status='completed_exact_replays',source_sha256=sha(__file__),protocol_sha256=sha(pp),original_receipt_sha256=sha(V/'bilinear_input_warmup_receipt.json'),rare_training_index=rare,rare_cache_index=1280,models=models,scope='Instrumentation-only replay; all2000losses and bothfinalparameter/Adamstates exactly reproduce original. Telemetry is posthoc observational; no optimizer variant or heldout evaluation.')
 dest=V/'bilinear_input_optimization_replay.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
