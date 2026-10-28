"""H899 diagnostic PC ordering and template covariance, fixed eight_AB fit."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
S='bilin18_output_pc_diagnosis'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','units':ART/'bilin18_number_unit_recovery_units.pt','fits':ART/'bilin18_fixed_axis_output_coverage_fits.pt','panel':ART/'bilin18_blind_number_unit_manifest.json','captures':ART/'bilin18_number_transfer_screen_captures.pt'};hashes={k:sha(p) for k,p in paths.items()};fits=torch.load(paths['fits'],weights_only=False);examples=[e for e in json.loads(paths['panel'].read_text())['examples'] if e['group']!='natural'];assert len(examples)==64;cfg=dict(experiment='H899',hashes=hashes,axis='original_A',fits=['eight_AB'],components=['mean','pc1','pc2','pc3','pc4','pc5','pc6','pc7','rank2','rank7'],design='Frozen eight_AB fit: mean, each individual PC with mean, cumulative rank2/7, native and opposite-number interventions plus exact/removal. Covariance decomposition on discovery. Diagnostic, no selected replacement unit. H89364 synthetic contexts: nouns and grammar templates absent from these output fits, reused diagnostic panel not globally untouched confirmation. No fit selection.');(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');U={k:v.cuda() for k,v in torch.load(paths['units'],weights_only=False)['original_A'].items()};model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 mod=model.blocks[12].mlp;L,R,D=[m.weight.double() for m in [mod.Left,mod.Right,mod.Down]];b=U['b'];origin=U['origin'];Q=((b@L.T)*(b@R.T))@D.T;compiled={}
 fit=fits['eight_AB'];mu=fit['mu'].cuda();basis=fit['basis'].cuda();panel=torch.load(paths['captures'],weights_only=False);ix=fit['indices'];raw=torch.stack([panel['states'][i][12] for i in ix]).double().cuda();u=raw/(raw.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();t=u@b-origin;w=u-t[:,None]*b;C=((w@L.T)*(b@R.T)+(w@R.T)*(b@L.T))@D.T;fmts=[panel['examples'][i]['format'] for i in ix];between=torch.stack([C[[j for j,f in enumerate(fmts) if f==fmt]].mean(0)-mu for fmt in fmts]);within=C-mu-between;covariance=[];total=(C-mu).square().sum();decomp_error=float(((C-mu).square().sum()-between.square().sum()-within.square().sum()).abs()/total);mean_error=float((mu-C.mean(0)).norm()/mu.norm())
 for j in range(7):
  v=basis[j];pv=((C-mu)@v).square().sum();bv=(between@v).square().sum();wv=(within@v).square().sum();covariance.append(dict(pc=j+1,total_variance_fraction=float(pv/total),between_template_fraction_of_PC=float(bv/pv),within_template_fraction_of_PC=float(wv/pv)))
 selections={'mean':[],**{'pc'+str(j+1):[j] for j in range(7)},'rank2':[0,1],'rank7':list(range(7))}
 for name,jj in selections.items():
  B=basis[jj];a=((B@D)*(R@b))@L+((B@D)*(L@b))@R;compiled[name]=(mu,B,a-(a@b)[:,None]*b,origin*(a@b)-B@mu)
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
 audits=dict(covariance_decomposition_relative=decomp_error,frozen_mean_replay_relative=mean_error,hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),identity_logits_max_abs=identity,compiled_encoder_relative=compiler,whole_model_forwards=actual,expected_forwards=64*(1+2*len(modes)),test_evaluations=0,finite=all(np.isfinite(r['KL']) and np.isfinite(r['exact_to_approx_KL']) for r in records));assert decomp_error<1e-10 and mean_error<1e-10 and audits['hashes_unchanged'] and identity<1e-4 and compiler<1e-10 and actual==audits['expected_forwards'] and audits['finite'];(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,covariance=covariance,between_template_total_fraction=float(between.square().sum()/total),records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
