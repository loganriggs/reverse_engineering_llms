"""H773 matched rank-eight reconstruction versus direct output-KL training pilot."""
import json,time,hashlib
from pathlib import Path
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_l5_refined_output_geometry import forward
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_l5_direct_kl_rank8'
class RankEight(torch.nn.Module):
 def __init__(self):
  super().__init__();self.encoder=torch.nn.Parameter(torch.eye(128)[:,:8].clone());self.decoder=torch.nn.Parameter(torch.eye(128)[:8].clone())
 def forward(self,z):return z@self.encoder@self.decoder

def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_l5_natural_featurizer_baselines_data.pt','base':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','previous':ART/'bilin18_l5_weighted_fits_replication.json','runner':Path(__file__),'forward_source':Path(__file__).with_name('run_bilin18_l5_refined_output_geometry.py'),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);base=torch.load(paths['base'],weights_only=False)['fits']['L5H5'];train=sorted(range(2048),key=lambda d:hashlib.sha256(f'H773:{d}'.encode()).hexdigest())[:128];val=list(dict.fromkeys(c['doc'] for c in json.loads(paths['previous'].read_text())['config']['validation_contexts']));cfg={'experiment':'H773','hashes':hashes,'train_documents':train,'validation_documents':val,'lengths':[32,120],'rank':8,'parameters':2048,'steps':128,'batch_size':8,'learning_rate':.001,'schedule_seed':7731,'design':'Shared rank8 linear bottleneck in frozen write coordinates, fixed natural mean/global scale, initialized at original write-PCA8. Both encoder and decoder train. Same initialization, minibatch schedule,128Adamsteps and LR for reconstruction and direct KL(native||edited). Normalize each training loss by its initial first-batch value. Teacher model frozen. Final checkpoint only, no validation selection.','limitations':'Single deterministic initialization/schedule, short pilot not convergence or tuned-objective comparison. Reused validation panel; no sparse gates, manifold or causal transfer claims. KL teacher may preserve biases/errors; observed-token CE separate.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval().requires_grad_(False)
 E=base['write_pca']['encode'].float().cuda();B=base['write_pca']['decode'].float().cuda();mu=base['mean'].float().cuda();large=(data['activations']['L5H5'][:2048].reshape(-1,128).double()-base['mean'])@base['write_pca']['encode'];scale=float(large.square().mean().sqrt());del large
 cache={};count=0;backwards=0;captureerr=[]
 with torch.no_grad():
  for n in cfg['lengths']:
   ds=train+val
   for lo in range(0,len(ds),8):
    bb=ds[lo:lo+8];saved={};out=forward(model,data['all_token_ids'][bb,:n].cuda(),capture=saved).cpu();count+=len(bb)
    for i,d in enumerate(bb):cache[d,n]={'head':saved['head'][i],'logits':out[i]};captureerr.append(float((saved['head'][i]-data['activations']['L5H5'][d,n//8-1]).abs().max()))
 def batch(ds,n):
  heads=torch.stack([cache[d,n]['head'] for d in ds]).cuda();teacher=torch.stack([cache[d,n]['logits'] for d in ds]).cuda().double().log_softmax(-1);return data['all_token_ids'][ds,:n].cuda(),heads,(heads-mu)@E/scale,teacher
 def kl(logits,teacher):return (teacher.exp()*(teacher-logits.double().log_softmax(-1))).sum(-1).mean()
 nets={name:RankEight().cuda() for name in ['initial','reconstruction','direct_KL']};gen=torch.Generator().manual_seed(7731);schedule=[(cfg['lengths'][i%2],[train[j] for j in torch.randint(len(train),(8,),generator=gen).tolist()]) for i in range(cfg['steps'])];n,ds=schedule[0];tokens,heads,z,teacher=batch(ds,n)
 # Audit end-to-end KL gradient through a learned encoder parameter direction.
 net=nets['direct_KL'];rec=net(z);loss=kl(forward(model,tokens,(rec*scale)@B+mu-heads),teacher);count+=len(ds);g=torch.autograd.grad(loss,net.encoder)[0];backwards+=len(ds);direction=g/g.norm();original=net.encoder.detach().clone();fd=[]
 for eps in [.001,.003]:
  values=[]
  for sign in [-1,1]:
   with torch.no_grad():net.encoder.copy_(original+sign*eps*direction);values.append(float(kl(forward(model,tokens,(net(z)*scale)@B+mu-heads),teacher)));count+=len(ds)
  fd.append({'epsilon':eps,'analytic':float(g.norm()),'numerical':(values[1]-values[0])/(2*eps),'relative_error':abs((values[1]-values[0])/(2*eps)-float(g.norm()))/float(g.norm())})
 with torch.no_grad():net.encoder.copy_(original)
 delta=torch.zeros_like(heads,requires_grad=True);zero_loss=kl(forward(model,tokens,delta),teacher);count+=len(ds);zg=torch.autograd.grad(zero_loss,delta)[0];backwards+=len(ds)
 with torch.no_grad():full=((z*scale)@B+mu-heads);fullout=forward(model,tokens,full);count+=len(ds);fullerr=float((fullout-torch.stack([cache[d,n]['logits'] for d in ds]).cuda()).abs().max());normalizers={'reconstruction':float((nets['initial'](z)-z).square().mean()),'direct_KL':float(loss.detach())}
 assert max(r['relative_error'] for r in fd)<.05 and abs(float(zero_loss))<1e-8 and float(zg.abs().max())<1e-7 and fullerr<1e-4
 print('AUDIT',fd,'identityKL',float(zero_loss),'gradient',float(zg.abs().max()),'full',fullerr,flush=True);logs=[]
 for name in ['reconstruction','direct_KL']:
  net=nets[name];opt=torch.optim.Adam(net.parameters(),lr=.001)
  for step,(n,ds) in enumerate(schedule):
   tokens,heads,z,teacher=batch(ds,n);opt.zero_grad();rec=net(z)
   if name=='reconstruction':objective=(rec-z).square().mean()
   else:objective=kl(forward(model,tokens,(rec*scale)@B+mu-heads),teacher);count+=len(ds)
   normalized=objective/normalizers[name];assert torch.isfinite(normalized);normalized.backward();backwards+=len(ds) if name=='direct_KL' else 0;opt.step()
   if step%16==0 or step==127:logs.append({'method':name,'step':step+1,'raw_loss':float(objective.detach())});print(logs[-1],flush=True)
 states={name:net.cpu().state_dict() for name,net in nets.items()};torch.save({'states':states,'scale':scale,'config':cfg,'normalizers':normalizers,'schedule':schedule},ART/f'{STEM}_fits.pt');fithash=sha(ART/f'{STEM}_fits.pt')
 for net in nets.values():net.cuda().eval()
 records=[]
 with torch.no_grad():
  for split,docs in [('train',train),('validation',val)]:
   for n in cfg['lengths']:
    for lo in range(0,len(docs),8):
     ds=docs[lo:lo+8];tokens,heads,z,teacher=batch(ds,n)
     for name,net in nets.items():
      rec=net(z);out=forward(model,tokens,(rec*scale)@B+mu-heads).double().log_softmax(-1);count+=len(ds)
      for i,d in enumerate(ds):target=int(data['all_token_ids'][d,n]);records.append({'split':split,'doc':d,'length':n,'method':name,'KL':float((teacher[i].exp()*(teacher[i]-out[i])).sum()),'CE_change':float(teacher[i,target]-out[i,target]),'write_MSE_scaled':float((rec[i]-z[i]).square().mean())})
 summary=[]
 for split in ['train','validation']:
  for name in nets:
   rr=[r for r in records if r['split']==split and r['method']==name];summary.append({'split':split,'method':name,**{k:sum(r[k] for r in rr)/len(rr) for k in ['KL','CE_change','write_MSE_scaled']}})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'fit_hash_unchanged':sha(ART/f'{STEM}_fits.pt')==fithash,'finite_difference':fd,'identity_KL':float(zero_loss),'identity_gradient_max_abs':float(zg.abs().max()),'full_rank_logit_replay_max_abs':fullerr,'cached_head_replay_max_abs':max(captureerr),'model_parameters_frozen':all(not p.requires_grad for mod in [model.embedding,model.blocks,model.lm_head] for p in mod.parameters()),'document_splits_disjoint':not bool(set(train)&set(val)),'whole_model_forwards':count,'KL_backward_context_evaluations':backwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['fit_hash_unchanged'] and audits['model_parameters_frozen'] and audits['document_splits_disjoint'] and max(captureerr)<1e-4
 (ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'summary':summary,'records':records,'training_log':logs,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audits},indent=2),flush=True)
if __name__=='__main__':main()
