"""H764 refined task-blind Monte Carlo output Fisher metric and train-only geometry."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_l5_refined_output_geometry'
def forward(model,ids,delta=None,capture=None):
 def edit(m,args):
  y=args[0].clone()
  if capture is not None:capture['head']=y[:,-1,640:768].detach().cpu()
  if delta is not None:y[:,-1,640:768]+=delta.to(y)
  return(y,)
 hook=model.blocks[5].attn.c_proj.register_forward_pre_hook(edit)
 try:
  x0=F.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None
  for block in model.blocks:x,bus=block(x,bus,x0)
  return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1]).float()/30)
 finally:hook.remove()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_l5_natural_featurizer_baselines_data.pt','baseline_fits':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py')};hashes={k:sha(p) for k,p in paths.items()};docids=sorted(range(2048),key=lambda i:hashlib.sha256(f'H763:{i}'.encode()).hexdigest())[:128];contexts=[{'doc':doc,'length':[16,32,64,128][i%4]} for i,doc in enumerate(docids)];cfg={'experiment':'H764','hashes':hashes,'contexts':contexts,'categorical_samples_per_context':64,'sample_seed':7631,'ridge_fractions':[.01,.1],'budgets':[8,16,32,64,128],'design':'Natural training contexts only, fixed hashed128documents and balanced4prefix lengths. Freeze model weights; differentiate sampled native log-probabilities wrt additive final-position L5H5 output. Average gradient outer products estimates local output-distribution Fisher in head coordinates. Compare first/second32sample halves. Ridge by .01/.1 times trace/128; train-centered PCA in resulting metric, using all32768natural training head vectors. Both metrics retained, no task-based choice. Audit native forward/capture replay, finite differences and metric algebra.','limitations':'Finite MonteCarlo local sensitivity, global average metric ignores context-dependent activation/sensitivity correlations. Final-position gradients across4lengths; natural covariance includes16positions. Quadratic local metric not actual large-intervention KL. No causal evaluation or task fitting in this experiment.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['data'],weights_only=False);ids=data['all_token_ids'];raw=data['activations']['L5H5'];model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval().requires_grad_(False)
 gen=torch.Generator(device='cuda').manual_seed(7631);buckets=defaultdict(list)
 for ctx in contexts:buckets[ctx['length']].append(ctx)
 grads=[];sample_records=[];replay=[];headreplay=[];fd=[];forwards=0;backwards=0
 for length,cc in buckets.items():
  for lo in range(0,len(cc),8):
   bb=cc[lo:lo+8];tokens=ids[[c['doc'] for c in bb],:length].cuda();reference,_=run(model,tokens);forwards+=len(bb)
   for block in model.blocks:block.attn.rotary.seq_len_cached=None
   delta=torch.zeros(len(bb),128,device='cuda',requires_grad=True);saved={};logits=forward(model,tokens,delta,saved);forwards+=len(bb);replay.append(float((logits.detach()-reference).abs().max()));expected=raw[[c['doc'] for c in bb],length//8-1];headreplay.append(float((saved['head']-expected).abs().max()));lp=logits.log_softmax(-1);samples=torch.multinomial(lp.detach().exp(),64,replacement=True,generator=gen);gg=[]
   for j in range(64):
    g=torch.autograd.grad(lp.gather(1,samples[:,j,None]).sum(),delta,retain_graph=j<63)[0];gg.append(g.detach().cpu().double());backwards+=len(bb)
   batchgrads=torch.stack(gg,1);grads.append(batchgrads)
   for i,c in enumerate(bb):sample_records.append({**c,'tokens':samples[i].cpu().tolist()})
   if lo==0:
    g0=batchgrads[:,0].float().cuda();direction=g0/g0.norm(dim=1,keepdim=True).clamp_min(1e-20);eps=.01
    with torch.no_grad():
     plus=forward(model,tokens,eps*direction).log_softmax(-1).gather(1,samples[:,0,None]).squeeze(1);minus=forward(model,tokens,-eps*direction).log_softmax(-1).gather(1,samples[:,0,None]).squeeze(1)
    numerical=(plus-minus)/(2*eps);analytic=g0.norm(dim=1);forwards+=2*len(bb)
    for i,c in enumerate(bb):fd.append({**c,'analytic':float(analytic[i]),'finite_difference':float(numerical[i]),'absolute_error':float((numerical[i]-analytic[i]).abs()),'relative_error':float((numerical[i]-analytic[i]).abs()/analytic[i].clamp_min(1e-12))})
   del logits,lp,delta
 print('GRADIENTS',forwards,backwards,flush=True);gg=torch.cat(grads);assert torch.isfinite(gg).all();flat=gg.reshape(-1,128);metric=flat.T@flat/len(flat);a=gg[:,:32].reshape(-1,128);b=gg[:,32:].reshape(-1,128);Ga=a.T@a/len(a);Gb=b.T@b/len(b);eig=torch.linalg.eigvalsh(metric);mu=raw[:2048].reshape(-1,128).double().mean(0);xc=raw[:2048].reshape(-1,128).double()-mu;vc=raw[2048:].reshape(-1,128).double()-mu;W=model.blocks[5].attn.c_proj.weight[:,640:768].detach().cpu().double();fits={};records=[];fit_audits={}
 for ridge in cfg['ridge_fractions']:
  G=metric+ridge*metric.trace()/128*torch.eye(128,dtype=torch.double);ev,U=torch.linalg.eigh(G);A=U*ev.sqrt();Ai=U.T/ev.sqrt()[:,None];y=xc@A;se,P=torch.linalg.eigh(y.T@y/len(y));P=P.flip(1);E=A@P;B=P.T@Ai;name=f'fisher_ridge_{ridge}';fits[name]={'mean':mu,'encode':E,'decode':B,'metric':G,'metric_eigenvalues':ev,'weighted_covariance_eigenvalues':se.flip(0)};fit_audits[name]={'inverse_max_abs':float((E@B-torch.eye(128)).abs().max()),'metric_factor_max_abs':float((A@A.T-G).abs().max()),'condition':float(ev.max()/ev.min())}
  for k in cfg['budgets']:
   err=vc-(vc@E[:,:k])@B[:k];records.append({'method':name,'budget':k,'validation_metric_relative_MSE':float((err@A).square().sum()/(vc@A).square().sum()),'validation_write_relative_MSE':float((err@W.T).square().sum()/(vc@W.T).square().sum()),'validation_coordinate_relative_MSE':float(err.square().sum()/vc.square().sum())})
 audits={'hashes_unchanged':{k:sha(p) for k,p in paths.items()}==hashes,'sample_contexts_train_only':all(c['doc']<2048 for c in contexts),'native_forward_replay_max_abs':max(replay),'natural_head_capture_replay_max_abs':max(headreplay),'finite_difference_max_relative_error':max(r['relative_error'] for r in fd),'finite_difference_max_absolute_error':max(r['absolute_error'] for r in fd),'finite_difference_pass':all(r['absolute_error']<5e-4+.03*abs(r['analytic']) for r in fd),'metric_min_eigenvalue':float(eig.min()),'metric_trace':float(metric.trace()),'effective_rank':float(metric.trace().square()/metric.square().sum()),'sample_half_relative_Frobenius_difference':float((Ga-Gb).norm()/metric.norm()),'sample_half_matrix_cosine':float((Ga*Gb).sum()/Ga.norm()/Gb.norm()),'fit':fit_audits,'finite_gradients':bool(torch.isfinite(gg).all()),'whole_model_forwards':forwards,'sample_logprob_backward_evaluations':backwards,'test_evaluations':0};print(audits,flush=True);assert audits['hashes_unchanged'] and audits['sample_contexts_train_only'] and audits['finite_gradients'] and audits['finite_difference_pass'] and max(replay)<1e-4 and max(headreplay)<1e-4 and min(eig)>-1e-10 and max(v['inverse_max_abs'] for v in fit_audits.values())<1e-10
 torch.save({'fits':fits,'metric':metric,'gradients':gg,'sample_records':sample_records},ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'finite_difference_checks':fd,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
if __name__=='__main__':main()
