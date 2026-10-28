"""H900 within-template PCA rule and shuffled-label controls."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
from run_bilin18_broad_task_competence import continuation_id
from audit_bilin18_mlp_discriminators import TOKENIZER
from transformers import AutoTokenizer
S='bilin18_within_template_rule'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','units':ART/'bilin18_number_unit_recovery_units.pt','fits':ART/'bilin18_fixed_axis_output_coverage_fits.pt','panel':ART/'bilin18_blind_number_unit_manifest.json','captures':ART/'bilin18_number_transfer_screen_captures.pt'};hashes={k:sha(p) for k,p in paths.items()};fits=torch.load(paths['fits'],weights_only=False);examples=[];excluded=[];nouns=[('baker','bakers'),('actor','actors'),('dancer','dancers'),('painter','painters'),('guard','guards'),('sailor','sailors'),('writer','writers'),('visitor','visitors')];cfg=dict(experiment='H900',hashes=hashes,axis='original_A',fit='eight_AB',rules=['pooled1','pooled2','within1','shuffle0','shuffle1','shuffle2','shuffle3'],seeds=[900,901,902,903],nouns=nouns,templates=['near the door','in the room'],design='Fixed template-residual covariance extraction rank1, same globalmean; pooled rank1/2 and four balanced endpoint-label shuffles. No numberlabels for output fit. New declared combinations, eligibility only, no correctness filtering. Labels used during discovery only.');(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 for pair,(a,b) in enumerate(nouns):
  rows=[];errors=[]
  for fmt,tail in [('I','near the door'),('J','in the room')]:
   for group in ['grammar','copy']:
    for order,noun in enumerate([a,b]):
     text=f'The {noun} {tail}' if group=='grammar' else (f'word {noun}; word {noun}; word' if fmt=='I' else f'item {noun}, item {noun}, item');answers=['is','are'] if group=='grammar' else [a,b];ids=[continuation_id(tok,text,' '+v) for v in answers]
     if any(v is None for v in ids):errors.append('answer_not_single');continue
     rows.append(dict(pair=pair,format=fmt,group=group,order=order,text=text,ids=tok.encode(text,add_special_tokens=False),answer_ids=ids))
  if errors:excluded.append(dict(pair=pair,reasons=errors))
  else:examples.extend(rows)
 assert len(examples)>0;(ART/f'{S}_manifest.json').write_text(json.dumps(dict(examples=examples,excluded=excluded),indent=2)+'\n');U={k:v.cuda() for k,v in torch.load(paths['units'],weights_only=False)['original_A'].items()};model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 mod=model.blocks[12].mlp;L,R,D=[m.weight.double() for m in [mod.Left,mod.Right,mod.Down]];b=U['b'];origin=U['origin'];Q=((b@L.T)*(b@R.T))@D.T;compiled={}
 fit=fits['eight_AB'];mu=fit['mu'].cuda();basis=fit['basis'].cuda();panel=torch.load(paths['captures'],weights_only=False);ix=fit['indices'];raw=torch.stack([panel['states'][i][12] for i in ix]).double().cuda();u=raw/(raw.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();t=u@b-origin;w=u-t[:,None]*b;C=((w@L.T)*(b@R.T)+(w@R.T)*(b@L.T))@D.T;fmts=[panel['examples'][i]['format'] for i in ix];between=torch.stack([C[[j for j,f in enumerate(fmts) if f==fmt]].mean(0)-mu for fmt in fmts]);within=C-mu-between;covariance=[];total=(C-mu).square().sum();decomp_error=float(((C-mu).square().sum()-between.square().sum()-within.square().sum()).abs()/total);mean_error=float((mu-C.mean(0)).norm()/mu.norm())
 for j in range(7):
  v=basis[j];pv=((C-mu)@v).square().sum();bv=(between@v).square().sum();wv=(within@v).square().sum();covariance.append(dict(pc=j+1,total_variance_fraction=float(pv/total),between_template_fraction_of_PC=float(bv/pv),within_template_fraction_of_PC=float(wv/pv)))
 _,_,Vwithin=torch.linalg.svd(within,full_matrices=False);bases={'pooled1':basis[:1],'pooled2':basis[:2],'within1':Vwithin[:1]};labels=torch.tensor([f=='B' for f in fmts],device='cuda');shuffle_labels={}
 for seed in range(900,904):
  gen=torch.Generator(device='cuda').manual_seed(seed);lab=labels[torch.randperm(len(labels),generator=gen,device='cuda')];res=C-torch.stack([C[lab==v].mean(0) for v in lab]);_,_,V=torch.linalg.svd(res,full_matrices=False);name='shuffle'+str(seed-900);bases[name]=V[:1];shuffle_labels[name]=lab.cpu().tolist()
 geometry={name:dict(abs_cosine_PC1=float(abs(B[0]@basis[0])),abs_cosine_PC2=float(abs(B[0]@basis[1]))) for name,B in bases.items()};torch.save(dict(mean=mu.cpu(),bases={k:v.cpu() for k,v in bases.items()},shuffle_labels=shuffle_labels),ART/f'{S}_fits.pt');fit_hash=sha(ART/f'{S}_fits.pt')
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
 audits=dict(frozen_output_fits_unchanged=sha(ART/f'{S}_fits.pt')==fit_hash,covariance_decomposition_relative=decomp_error,frozen_mean_replay_relative=mean_error,hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),identity_logits_max_abs=identity,compiled_encoder_relative=compiler,whole_model_forwards=actual,expected_forwards=len(examples)*(1+2*len(modes)),test_evaluations=0,finite=all(np.isfinite(r['KL']) and np.isfinite(r['exact_to_approx_KL']) for r in records));assert audits['frozen_output_fits_unchanged'] and decomp_error<1e-10 and mean_error<1e-10 and audits['hashes_unchanged'] and identity<1e-4 and compiler<1e-10 and actual==audits['expected_forwards'] and audits['finite'];(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,examples=examples,excluded=excluded,geometry=geometry,covariance=covariance,between_template_total_fraction=float(between.square().sum()/total),records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
