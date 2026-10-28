"""H905 paired-change versus pooled PCA; finite L12 MLP interventions."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict
import json,time,torch,numpy as np
BASE='bilin18_matched_context_operator_test'
@torch.inference_mode()
def run_layer(layer):
 S=BASE+'_l'+str(layer)
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','panel':ART/'bilin18_demonstrated_language_contexts_captures.pt','means':ART/f'bilin18_language_shared_partner_l{layer}_fits.pt','screen':ART/'bilin18_demonstrated_language_contexts.json'};hashes={k:sha(p) for k,p in paths.items()};cfg=dict(experiment='H915',hashes=hashes,layer=layer,design='Matched query panel: first4oldpairs discovery, remainingold/newpairs transfer120rows. Percondition task/copy means frozen before eval; both maps and their average independently applied to every context. Legacyunmatched means controls. N0/N2 passcompetence,N1fails; retainall. Conditions chosen by baseline screening, not untouchedconfirmation.');(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');panel=torch.load(paths['panel'],weights_only=False);ex=panel['examples'];uall=torch.stack([panel['normalized_states'][i][layer] for i in range(len(ex))]).double().cuda();matched={(e['family'],e['format'],e['pair'],e['order']):i for i,e in enumerate(ex)};evalix=[i for i,e in enumerate(ex) if e['pair']>=4];assert len(evalix)==120;loaded=torch.load(paths['means'],weights_only=False)['means'];means={'legacy_task':loaded['task_pooled'].cuda(),'legacy_copy':((loaded['copy','A']+loaded['copy','B'])/2).cuda()};fit_indices={}
 for family in ['task','copy']:
  for fmt in ['N0','N1','N2']:
   ix=[i for i,e in enumerate(ex) if e['pair']<4 and e['family']==family and e['format']==fmt];assert len(ix)==8;fit_indices[family+'_'+fmt]=ix;means[family,fmt]=uall[ix].mean(0)
 torch.save({k:v.cpu() for k,v in means.items()},ART/f'{S}_means.pt');fit_hash=sha(ART/f'{S}_means.pt');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 mod=model.blocks[layer].mlp;L,R,D=[m.weight.double() for m in [mod.Left,mod.Right,mod.Down]];state={'mode':'baseline'};cacheerr=0.;stash={}
 geometry=[]
 def operator(mu):return D@((R@mu)[:,None]*L+(L@mu)[:,None]*R)
 legacy=operator(means['legacy_task']-means['legacy_copy'])
 for fmt in ['N0','N1','N2']:
  ta,co=operator(means['task',fmt]),operator(means['copy',fmt]);delta=ta-co;geometry.append(dict(format=fmt,task_copy_operator_cosine=float((ta*co).sum()/ta.norm()/co.norm()),contrast_relative_to_task=float(delta.norm()/ta.norm()),contrast_relative_to_legacy=float(delta.norm()/legacy.norm()),contrast_legacy_cosine=float((delta*legacy).sum()/delta.norm()/legacy.norm())))
 def hook(mod,args,out):
  nonlocal cacheerr
  if state['mode']=='baseline':return
  u=args[0][:,-1].double();ix=state['indices'];donors=[matched[ex[i]['family'],ex[i]['format'],ex[i]['pair'],1-ex[i]['order']] for i in ix];delta=uall[donors]-u;cacheerr=max(cacheerr,float((u-uall[ix]).norm()/u.norm()));mode=state['mode']
  v=delta;linear=((v@L.T)*(u@R.T)+(u@L.T)*(v@R.T))@D.T;quadratic=((v@L.T)*(v@R.T))@D.T;exact=linear+quadratic
  if mode=='full':change=exact
  elif mode=='identity':change=torch.zeros_like(exact)
  elif mode=='native_linear':change=linear
  elif mode=='quadratic':change=quadratic
  else:
   key,term=mode.rsplit('_',1)
   if key.startswith('legacy'):partner=means[key].expand_as(u)
   elif key=='average':partner=torch.stack([(means['task',ex[i]['format']]+means['copy',ex[i]['format']])/2 for i in ix])
   else:partner=torch.stack([means['task' if key=='matched_task' else 'copy',ex[i]['format']] for i in ix])
   change=((v@L.T)*(partner@R.T)+(partner@L.T)*(v@R.T))@D.T
   if term=='quad':change=change+quadratic
  stash['capture']=torch.ones(len(ix));stash['radius_ratio']=((u+v).norm(dim=-1)/u.norm(dim=-1)).cpu();stash['write_error']=((change-exact).norm(dim=-1)/exact.norm(dim=-1)).cpu();z=out.clone();z[:,-1]+=change.to(out);return z
 handle=mod.register_forward_hook(hook);modes=['identity','full','native_linear','quadratic']+[key+'_linear' for key in ['matched_task','matched_copy','average','legacy_task','legacy_copy']];buckets=defaultdict(list)
 for i in evalix:buckets[len(ex[i]['ids'])].append(i)
 records=[];actual=0;identity=0.;baseline_replay=0.;refs={}
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),8):
    ix=ii[st:st+8];tokens=torch.tensor([ex[i]['ids'] for i in ix],device='cuda');state.update(mode='baseline',indices=ix);base=final_logits(model,tokens).double().cpu();actual+=len(ix);baseline_replay=max(baseline_replay,max(float((base[n]-panel['native'][i]).abs().max()) for n,i in enumerate(ix)))
    for mode in modes:
     state['mode']=mode;y=final_logits(model,tokens).double().cpu();actual+=len(ix)
     for n,i in enumerate(ix):
      e=ex[i];ids=e['answer_ids'];sign=1 if e['expected']==0 else -1;lp=torch.log_softmax(base[n],-1);lq=torch.log_softmax(y[n],-1)
      if mode=='identity':identity=max(identity,float((y[n]-base[n]).abs().max()))
      if mode=='full':refs[i]=lq.clone()
      r=dict(index=i,cohort=e['cohort'],family=e['family'],format=e['format'],pair=e['pair'],mode=mode,KL=float((lp.exp()*(lp-lq)).sum()),native_margin=sign*float(base[n,ids[0]]-base[n,ids[1]]),margin=sign*float(y[n,ids[0]]-y[n,ids[1]]),native_top1_correct=int(base[n].argmax())==ids[e['expected']],write_relative_error=float(stash['write_error'][n]),radius_ratio=float(stash['radius_ratio'][n]))
      if mode!='identity':ref=refs[i];r['full_to_approx_KL']=float((ref.exp()*(ref-lq)).sum());r['full_to_native_KL']=float((ref.exp()*(ref-lp)).sum())
      records.append(r)
   print('DONE',length,actual,flush=True)
 finally:handle.remove()
 audits=dict(newmeans_unchanged=sha(ART/f'{S}_means.pt')==fit_hash,hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),cached_activation_relative=cacheerr,identity_logits_max_abs=identity,baseline_logits_replay_max_abs=baseline_replay,whole_model_forwards=actual,expected_forwards=1200,test_evaluations=0,finite=all(np.isfinite(r['KL']) for r in records));assert audits['newmeans_unchanged'] and audits['hashes_unchanged'] and cacheerr<1e-5 and identity<1e-4 and baseline_replay<1e-3 and actual==1200 and audits['finite'];(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,geometry=geometry,fit_indices=fit_indices,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
def main():
 for layer in [13,14]:run_layer(layer)
 parts=[json.loads((ART/f'{BASE}_l{k}.json').read_text()) for k in [13,14]];records=[dict(r,layer=k) for k,p in zip([13,14],parts) for r in p['records']];audits=dict(hashes_unchanged=all(p['audits']['hashes_unchanged'] for p in parts),cached_activation_relative=max(p['audits']['cached_activation_relative'] for p in parts),identity_logits_max_abs=max(p['audits']['identity_logits_max_abs'] for p in parts),baseline_logits_replay_max_abs=max(p['audits']['baseline_logits_replay_max_abs'] for p in parts),whole_model_forwards=sum(p['audits']['whole_model_forwards'] for p in parts),test_evaluations=0,finite=all(p['audits']['finite'] for p in parts));cfg=dict(experiment='H915',layers=[13,14],scope='Matched query means, discovery4pairs, transferold/new; N0/N2competent,N1failed, allretained. No layer/map selection from effects.',layer_configs=[p['config'] for p in parts]);(ROOT/f'configs/{BASE}.json').write_text(json.dumps(cfg,indent=2)+'\n');(ART/f'{BASE}.json').write_text(json.dumps(dict(config=cfg,geometry=[dict(r,layer=k) for k,p in zip([13,14],parts) for r in p['geometry']],records=records,audits=audits),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
