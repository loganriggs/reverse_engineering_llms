"""H901 frozen rule replay and discovery-only context grouping."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
S='bilin18_context_group_stress'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','units':ART/'bilin18_number_unit_recovery_units.pt','fits':ART/'bilin18_fixed_axis_output_coverage_fits.pt','panel':ART/'bilin18_blind_number_unit_manifest.json','captures':ART/'bilin18_number_transfer_screen_captures.pt','rules':ART/'bilin18_within_template_rule_fits.pt'};hashes={k:sha(p) for k,p in paths.items()};fits=torch.load(paths['fits'],weights_only=False);examples=[e for e in json.loads(paths['panel'].read_text())['examples'] if e['group']!='natural'];assert len(examples)==64;cfg=dict(experiment='H902',hashes=hashes,axis='original_A',fits=['eight_ABCDEF'],cluster_counts=[2,4,6,12],components=['pooled1','pooled2','pooled7','within1','cluster_input','cluster_output'],design='Six-template discovery: K2/4/6/12 Lloyd grouping in normalized input or output writes, deterministic farthest-pair then farthest-point initialization. Discovery only, max100iterations; failed clusterings retained in diagnostics and not evaluated. No label-based selection. K values fixed rather than inferred. Labels used for diagnostic metrics and labeled within-template reference only. H89364 synthetic contexts: nouns and grammar templates absent from these output fits, reused diagnostic panel not globally untouched confirmation. No fit selection.');(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');U={k:v.cuda() for k,v in torch.load(paths['units'],weights_only=False)['original_A'].items()};model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 mod=model.blocks[12].mlp;L,R,D=[m.weight.double() for m in [mod.Left,mod.Right,mod.Down]];b=U['b'];origin=U['origin'];Q=((b@L.T)*(b@R.T))@D.T;compiled={}
 fit=fits['eight_ABCDEF'];mu=fit['mu'].cuda();basis=fit['basis'].cuda();panel=torch.load(paths['captures'],weights_only=False);ix=fit['indices'];raw=torch.stack([panel['states'][i][12] for i in ix]).double().cuda();u=raw/(raw.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();t=u@b-origin;w=u-t[:,None]*b;C=((w@L.T)*(b@R.T)+(w@R.T)*(b@L.T))@D.T;fmts=[panel['examples'][i]['format'] for i in ix];between=torch.stack([C[[j for j,f in enumerate(fmts) if f==fmt]].mean(0)-mu for fmt in fmts]);within=C-mu-between;covariance=[];total=(C-mu).square().sum();decomp_error=float(((C-mu).square().sum()-between.square().sum()-within.square().sum()).abs()/total);mean_error=float((mu-C.mean(0)).norm()/mu.norm())
 for j in range(7):
  v=basis[j];pv=((C-mu)@v).square().sum();bv=(between@v).square().sum();wv=(within@v).square().sum();covariance.append(dict(pc=j+1,total_variance_fraction=float(pv/total),between_template_fraction_of_PC=float(bv/pv),within_template_fraction_of_PC=float(wv/pv)))
 _,_,WV=torch.linalg.svd(within,full_matrices=False);bases={'pooled1':basis[:1],'pooled2':basis[:2],'pooled7':basis[:7],'within1':WV[:1]};groupinfo=[];number=torch.tensor([panel['examples'][i]['order'] for i in ix],device='cuda');targets={'template':torch.tensor([ord(f)-ord('A') for f in fmts],device='cuda'),'number':number,'noun':torch.tensor([panel['examples'][i]['pair'] for i in ix],device='cuda')};delta=C[number==1].mean(0)-C[number==0].mean(0)
 def information(labs,y):
  n=len(y);entropy=0.;mi=0.
  for v in y.unique():
   py=float((y==v).sum())/n;entropy-=py*np.log(py)
   for k in labs.unique():
    pk=float((labs==k).sum())/n;p=float(((y==v)&(labs==k)).sum())/n
    if p:mi+=p*np.log(p/(py*pk))
  return mi/entropy
 for space,Z in [('input',u),('output',C)]:
  for K in [2,4,6,12]:
   name='cluster_'+space+'_'+str(K);dist=torch.cdist(Z,Z).square();idx=int(dist.argmax());chosen=[idx//len(Z),idx%len(Z)]
   while len(chosen)<K:chosen.append(int(dist[:,chosen].min(1).values.argmax()))
   centers=Z[chosen].clone();last=None;converged=False;failure=None
   for iteration in range(100):
    labs=torch.cdist(Z,centers).argmin(1)
    if len(labs.unique())!=K:failure='empty_cluster';break
    if last is not None and torch.equal(labs,last):converged=True;break
    centers=torch.stack([Z[labs==k].mean(0) for k in range(K)]);last=labs.clone()
   if not converged:groupinfo.append(dict(rule=name,converged=False,failure=failure or 'iteration_limit'));continue
   residual=C-torch.stack([C[labs==v].mean(0) for v in labs]);_,_,Vr=torch.linalg.svd(residual,full_matrices=False);bases[name]=Vr[:1];rd=residual[number==1].mean(0)-residual[number==0].mean(0);groupinfo.append(dict(rule=name,K=K,counts=[int((labs==k).sum()) for k in range(K)],labels=labs.cpu().tolist(),iterations=iteration+1,converged=True,label_entropy_explained={label:information(labs,y) for label,y in targets.items()},number_centroid_contrast_remaining_norm_ratio=float(rd.norm()/delta.norm()),abs_cosine_within1=float(abs(Vr[0]@bases['within1'][0]))))
 torch.save(dict(mean=mu.cpu(),bases={k:v.cpu() for k,v in bases.items()},groupinfo=groupinfo),ART/f'{S}_fits.pt');fit_hash=sha(ART/f'{S}_fits.pt')
 for name,B in bases.items():
  a=((B@D)*(R@b))@L+((B@D)*(L@b))@R;compiled[name]=(mu,B,a-(a@b)[:,None]*b,origin*(a@b)-B@mu)
 modes=['exact','drop_cross']+list(compiled);state={'mode':'baseline'};compiler=0.
 def hook(mod,args,out):
  nonlocal compiler
  if state['mode']=='baseline':return
  u=args[0][:,-1].double();t=u@b-origin;w=u-t[:,None]*b;C=((w@L.T)*(b@R.T)+(w@R.T)*(b@L.T))@D.T;mode=state['mode']
  if mode=='exact':hat=C
  elif mode=='drop_cross':hat=torch.zeros_like(C)
  else:
   mu,B,E,off=compiled[mode];hat=mu+(u@E.T+off)@B;ref=mu+(C-mu)@B.T@B;compiler=max(compiler,float((hat-ref).norm()/ref.norm()))
  tp=U['cc'][1-state['orders']]+U['mean_projection']-origin if state['flip'] else t;change=tp[:,None]*hat-t[:,None]*C+(tp.square()-t.square())[:,None]*Q;z=out.clone();z[:,-1]+=change.to(out);return z
 handle=mod.register_forward_hook(hook);buckets=defaultdict(list)
 for i,e in enumerate(examples):buckets[len(e['ids'])].append(i)
 records=[];actual=0;identity=0.;refs={}
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),8):
    ix=ii[st:st+8];tokens=torch.tensor([examples[i]['ids'] for i in ix],device='cuda');state.update(mode='baseline',orders=torch.tensor([examples[i]['order'] for i in ix],device='cuda'));base=final_logits(model,tokens).double().cpu();actual+=len(ix)
    for mode in modes:
     for flip in [False,True]:
      state.update(mode=mode,flip=flip);y=final_logits(model,tokens).double().cpu();actual+=len(ix)
      for n,i in enumerate(ix):
       e=examples[i];lp=torch.log_softmax(base[n],-1);lq=torch.log_softmax(y[n],-1);ids=e['answer_ids'];sign=1 if e['order']==0 else -1
       if mode=='exact':refs[i,flip]=lq.clone()
       if mode=='exact' and not flip:identity=max(identity,float((y[n]-base[n]).abs().max()))
       ref=refs[i,flip];records.append(dict(index=i,group=e['group'],format=e['format'],pair=e['pair'],order=e['order'],mode=mode,flip=flip,KL=float((lp.exp()*(lp-lq)).sum()),exact_to_approx_KL=float((ref.exp()*(ref-lq)).sum()),margin=sign*float(y[n,ids[0]]-y[n,ids[1]]),native_margin=sign*float(base[n,ids[0]]-base[n,ids[1]]),native_top1_correct=int(base[n].argmax())==ids[e['order']],edited_top1_correct=int(y[n].argmax())==ids[e['order']],top1_changed=int(base[n].argmax())!=int(y[n].argmax())))
   print('DONE',length,actual,flush=True)
 finally:handle.remove()
 audits=dict(newfits_unchanged=sha(ART/f'{S}_fits.pt')==fit_hash,covariance_decomposition_relative=decomp_error,frozen_mean_replay_relative=mean_error,hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),identity_logits_max_abs=identity,compiled_encoder_relative=compiler,whole_model_forwards=actual,expected_forwards=64*(1+2*len(modes)),test_evaluations=0,finite=all(np.isfinite(r['KL']) and np.isfinite(r['exact_to_approx_KL']) for r in records));assert audits['newfits_unchanged'] and decomp_error<1e-10 and mean_error<1e-10 and audits['hashes_unchanged'] and identity<1e-4 and compiler<1e-10 and actual==audits['expected_forwards'] and audits['finite'];(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,groupinfo=groupinfo,covariance=covariance,between_template_total_fraction=float(between.square().sum()/total),records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
