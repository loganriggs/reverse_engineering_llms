"""H901 frozen rule replay and discovery-only context grouping."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
S='bilin18_retained_group_means'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','units':ART/'bilin18_number_unit_recovery_units.pt','fits':ART/'bilin18_fixed_axis_output_coverage_fits.pt','panel':ART/'bilin18_blind_number_unit_manifest.json','captures':ART/'bilin18_number_transfer_screen_captures.pt','rules':ART/'bilin18_context_group_stress_fits.pt'};hashes={k:sha(p) for k,p in paths.items()};fits=torch.load(paths['fits'],weights_only=False);examples=[e for e in json.loads(paths['panel'].read_text())['examples'] if e['group']!='natural'];assert len(examples)==64;cfg=dict(experiment='H903',hashes=hashes,axis='original_A',fits=['eight_ABCDEF'],cluster_counts=[2,4,6,12],components=['global_residual','cluster_mean','conditional_live','conditional_fixed','flat_Kplus1'],design='Retain H902 input cluster means, shared residualrank1. K2/4/6/12 frozen. Runtime nearestinputcentroid without tasklabels. Live routes recomputed on intervened normalized coordinate; fixed routes diagnostic isolates mean restoration. Flat rankK+1 approximately matches stored inference coefficients incl routingcentroids. Reused64 H893 contexts; no selection or freshconfirmation claim.');(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');U={k:v.cuda() for k,v in torch.load(paths['units'],weights_only=False)['original_A'].items()};model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 mod=model.blocks[12].mlp;L,R,D=[m.weight.double() for m in [mod.Left,mod.Right,mod.Down]];b=U['b'];origin=U['origin'];Q=((b@L.T)*(b@R.T))@D.T;compiled={}
 fit=fits['eight_ABCDEF'];mu=fit['mu'].cuda();basis=fit['basis'].cuda();panel=torch.load(paths['captures'],weights_only=False);ix=fit['indices'];raw=torch.stack([panel['states'][i][12] for i in ix]).double().cuda();u=raw/(raw.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();t=u@b-origin;w=u-t[:,None]*b;C=((w@L.T)*(b@R.T)+(w@R.T)*(b@L.T))@D.T;fmts=[panel['examples'][i]['format'] for i in ix];between=torch.stack([C[[j for j,f in enumerate(fmts) if f==fmt]].mean(0)-mu for fmt in fmts]);within=C-mu-between;covariance=[];total=(C-mu).square().sum();decomp_error=float(((C-mu).square().sum()-between.square().sum()-within.square().sum()).abs()/total);mean_error=float((mu-C.mean(0)).norm()/mu.norm())
 for j in range(7):
  v=basis[j];pv=((C-mu)@v).square().sum();bv=(between@v).square().sum();wv=(within@v).square().sum();covariance.append(dict(pc=j+1,total_variance_fraction=float(pv/total),between_template_fraction_of_PC=float(bv/pv),within_template_fraction_of_PC=float(wv/pv)))
 saved=torch.load(paths['rules'],weights_only=False);groupinfo=[];routers={};_,_,fullbasis=torch.linalg.svd(C-mu,full_matrices=False);compiled={};specs={};counts={}
 for K in [2,4,6,12]:
  info=next(r for r in saved['groupinfo'] if r['rule']=='cluster_input_'+str(K));labs=torch.tensor(info['labels'],device='cuda');centers=torch.stack([u[labs==k].mean(0) for k in range(K)]);means=torch.stack([C[labs==k].mean(0) for k in range(K)]);assert torch.equal(torch.cdist(u,centers).argmin(1),labs);B=saved['bases']['cluster_input_'+str(K)].cuda();routers[K]=(centers,means)
  for kind in ['global_residual','cluster_mean','conditional_live','conditional_fixed','flat']:
   name=kind+'_'+str(K);BB=fullbasis[:K+1] if kind=='flat' else (B[:0] if kind=='cluster_mean' else B);aa=((BB@D)*(R@b))@L+((BB@D)*(L@b))@R;E=aa-(aa@b)[:,None]*b;off=origin*(aa@b);compiled[name]=(BB,E,off);specs[name]=(kind,K);counts[name]=(2*(K+1)+1)*1152+(K+1) if kind=='flat' else ((2*K+2)*1152+K+1 if kind.startswith('conditional') else (2*K*1152+K if kind=='cluster_mean' else 3*1152+1))
  groupinfo.append(dict(K=K,counts=info['counts']))
 torch.save(dict(routers={k:(v[0].cpu(),v[1].cpu()) for k,v in routers.items()},compiled={k:tuple(v.cpu() for v in val) for k,val in compiled.items()},specs=specs,coefficient_counts=counts),ART/f'{S}_fits.pt');fit_hash=sha(ART/f'{S}_fits.pt');route_records=[]
 modes=['exact','drop_cross']+list(compiled);state={'mode':'baseline'};compiler=0.
 def hook(mod,args,out):
  nonlocal compiler
  if state['mode']=='baseline':return
  u=args[0][:,-1].double();t=u@b-origin;w=u-t[:,None]*b;C=((w@L.T)*(b@R.T)+(w@R.T)*(b@L.T))@D.T;mode=state['mode']
  tp=U['cc'][1-state['orders']]+U['mean_projection']-origin if state['flip'] else t;up=w+tp[:,None]*b
  if mode=='exact':hat=C
  elif mode=='drop_cross':hat=torch.zeros_like(C)
  else:
   kind,K=specs[mode];B,E,off=compiled[mode];center=mu
   if kind in ['cluster_mean','conditional_live','conditional_fixed']:
    centers,means=routers[K];rn=torch.cdist(u,centers).argmin(1);rf=torch.cdist(up,centers).argmin(1);route=rn if kind=='conditional_fixed' else rf;center=means[route]
    if state['flip']:
     for n,i in enumerate(state['indices']):route_records.append(dict(index=i,mode=mode,native_route=int(rn[n]),flipped_route=int(rf[n])))
   hat=center+(up@E.T+off-center@B.T)@B;ref=center+(C-center)@B.T@B;compiler=max(compiler,float((hat-ref).norm()/ref.norm()))
  change=tp[:,None]*hat-t[:,None]*C+(tp.square()-t.square())[:,None]*Q;z=out.clone();z[:,-1]+=change.to(out);return z
 handle=mod.register_forward_hook(hook);buckets=defaultdict(list)
 for i,e in enumerate(examples):buckets[len(e['ids'])].append(i)
 records=[];actual=0;identity=0.;refs={}
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),8):
    ix=ii[st:st+8];tokens=torch.tensor([examples[i]['ids'] for i in ix],device='cuda');state.update(mode='baseline',indices=ix,orders=torch.tensor([examples[i]['order'] for i in ix],device='cuda'));base=final_logits(model,tokens).double().cpu();actual+=len(ix)
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
 audits=dict(newfits_unchanged=sha(ART/f'{S}_fits.pt')==fit_hash,covariance_decomposition_relative=decomp_error,frozen_mean_replay_relative=mean_error,hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),identity_logits_max_abs=identity,compiled_encoder_relative=compiler,whole_model_forwards=actual,expected_forwards=64*(1+2*len(modes)),test_evaluations=0,finite=all(np.isfinite(r['KL']) and np.isfinite(r['exact_to_approx_KL']) for r in records));assert audits['newfits_unchanged'] and decomp_error<1e-10 and mean_error<1e-10 and audits['hashes_unchanged'] and identity<1e-4 and compiler<1e-10 and actual==audits['expected_forwards'] and audits['finite'];(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,coefficient_counts=counts,route_records=route_records,groupinfo=groupinfo,covariance=covariance,between_template_total_fraction=float(between.square().sum()/total),records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
