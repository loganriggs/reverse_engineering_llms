"""H905 paired-change versus pooled PCA; finite L12 MLP interventions."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict
import json,time,torch,numpy as np
BASE='bilin18_language_shared_partner'
@torch.inference_mode()
def run_layer(layer):
 S=BASE+'_l'+str(layer)
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','panel':ART/'bilin18_multidomain_matched_competence_captures.pt','mapping':ART/'bilin18_partner_subspace_fidelity.json','states':ART/'bilin18_partner_subspace_fidelity_native_mlp_states.pt'};hashes={k:sha(p) for k,p in paths.items()};cfg=dict(experiment='H908',hashes=hashes,layer=layer,design='Full pairedsource changes retained. Discovery contextmeans from first4pairs; evaluate last4pairs task/copy. Compare native linear, quadratic only, and ownfamily-template/taskpooled/oppositefamily-template/allpooled means with or without quadratic term. Meanlinear is midpoint-partner approximation; meanplusquadratic is recipient-partner approximation. No compression claim, reused screened layers/panel.');(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');panel=torch.load(paths['panel'],weights_only=False);mapping=json.loads(paths['mapping'].read_text());ex=[panel['examples'][i] for i in mapping['selection_original_indices']];states=torch.load(paths['states'],weights_only=False);raw=torch.stack([states[i,(layer,'raw')] for i in range(len(ex))]).double().cuda();uall=raw/(raw.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();matched={(e['domain'],e['family'],e['format'],e['pair'],e['order']):i for i,e in enumerate(ex)};fits={};evalix=[];meta=[];orth=0.;gen=torch.Generator(device='cuda').manual_seed(905)
 dom='language';pairs=sorted({e['pair'] for e in ex if e['domain']==dom});evalix=[i for i,e in enumerate(ex) if e['domain']==dom and e['pair'] in pairs[4:]];means={}
 for family in ['task','copy']:
  for fmt in ['A','B']:
   ix=[i for i,e in enumerate(ex) if e['domain']==dom and e['family']==family and e['format']==fmt and e['pair'] in pairs[:4]];assert len(ix)==8;means[family,fmt]=uall[ix].mean(0);meta.append(dict(family=family,format=fmt,discovery_indices=ix))
 means['task_pooled']=(means['task','A']+means['task','B'])/2;means['all_pooled']=torch.stack([means[f,t] for f in ['task','copy'] for t in ['A','B']]).mean(0);fits={'means':means}
 assert len(evalix)==32;torch.save({k:{n:v.cpu() for n,v in f.items()} for k,f in fits.items()},ART/f'{S}_fits.pt');fit_hash=sha(ART/f'{S}_fits.pt');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 mod=model.blocks[layer].mlp;L,R,D=[m.weight.double() for m in [mod.Left,mod.Right,mod.Down]];state={'mode':'baseline'};cacheerr=0.;stash={}
 def hook(mod,args,out):
  nonlocal cacheerr
  if state['mode']=='baseline':return
  u=args[0][:,-1].double();ix=state['indices'];donors=[matched[ex[i]['domain'],ex[i]['family'],ex[i]['format'],ex[i]['pair'],1-ex[i]['order']] for i in ix];delta=uall[donors]-u;cacheerr=max(cacheerr,float((u-uall[ix]).norm()/u.norm()));mode=state['mode']
  v=delta;linear=((v@L.T)*(u@R.T)+(u@L.T)*(v@R.T))@D.T;quadratic=((v@L.T)*(v@R.T))@D.T;exact=linear+quadratic
  if mode=='full':change=exact
  elif mode=='identity':change=torch.zeros_like(exact)
  elif mode=='native_linear':change=linear
  elif mode=='quadratic':change=quadratic
  else:
   key,term=mode.rsplit('_',1);pp=[]
   for i in ix:
    e=ex[i]
    if key=='own':p=means[e['family'],e['format']]
    elif key=='opposite':p=means['copy' if e['family']=='task' else 'task',e['format']]
    else:p=means[key]
    pp.append(p)
   partner=torch.stack(pp);change=((v@L.T)*(partner@R.T)+(partner@L.T)*(v@R.T))@D.T
   if term=='quad':change=change+quadratic
  stash['capture']=torch.ones(len(ix));stash['radius_ratio']=((u+v).norm(dim=-1)/u.norm(dim=-1)).cpu();stash['write_error']=((change-exact).norm(dim=-1)/exact.norm(dim=-1)).cpu();z=out.clone();z[:,-1]+=change.to(out);return z
 handle=mod.register_forward_hook(hook);modes=['identity','full','native_linear','quadratic']+[key+'_'+term for key in ['own','task_pooled','opposite','all_pooled'] for term in ['linear','quad']];buckets=defaultdict(list)
 for i in evalix:buckets[len(ex[i]['ids'])].append(i)
 records=[];actual=0;identity=0.;refs={}
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),8):
    ix=ii[st:st+8];tokens=torch.tensor([ex[i]['ids'] for i in ix],device='cuda');state.update(mode='baseline',indices=ix);base=final_logits(model,tokens).double().cpu();actual+=len(ix)
    for mode in modes:
     state['mode']=mode;y=final_logits(model,tokens).double().cpu();actual+=len(ix)
     for n,i in enumerate(ix):
      e=ex[i];assert all(len(g)==1 for g in e['groups']);ids=[g[0] for g in e['groups']];sign=1 if e['expected']==0 else -1;lp=torch.log_softmax(base[n],-1);lq=torch.log_softmax(y[n],-1)
      if mode=='identity':identity=max(identity,float((y[n]-base[n]).abs().max()))
      if mode=='full':refs[i]=lq.clone()
      r=dict(index=i,domain=e['domain'],family=e['family'],format=e['format'],pair=e['pair'],mode=mode,KL=float((lp.exp()*(lp-lq)).sum()),native_margin=sign*float(base[n,ids[0]]-base[n,ids[1]]),margin=sign*float(y[n,ids[0]]-y[n,ids[1]]),native_top1_correct=int(base[n].argmax())==ids[e['expected']],write_relative_error=float(stash['write_error'][n]),radius_ratio=float(stash['radius_ratio'][n]))
      if mode!='identity':ref=refs[i];r['full_to_approx_KL']=float((ref.exp()*(ref-lq)).sum())
      records.append(r)
   print('DONE',length,actual,flush=True)
 finally:handle.remove()
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),fits_unchanged=sha(ART/f'{S}_fits.pt')==fit_hash,orthonormal_error=orth,cached_activation_relative=cacheerr,identity_logits_max_abs=identity,whole_model_forwards=actual,expected_forwards=416,test_evaluations=0,finite=all(np.isfinite(r['KL']) for r in records));assert audits['hashes_unchanged'] and audits['fits_unchanged'] and orth<1e-10 and cacheerr<1e-5 and identity<1e-4 and actual==416 and audits['finite'];(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,fitmeta=meta,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
def main():
 for layer in [12,13,14]:run_layer(layer)
 parts=[json.loads((ART/f'{BASE}_l{k}.json').read_text()) for k in [12,13,14]];records=[dict(r,layer=k) for k,p in zip([12,13,14],parts) for r in p['records']];prior=json.loads((ART/'bilin18_multidomain_mlp_localization.json').read_text());lookup={(r['index'],r['mode']):r for r in prior['records']};replay=max(abs(r['margin']-lookup[r['index'],'layer_'+str(r['layer'])]['margin']) for r in records if r['mode']=='full');audits=dict(hashes_unchanged=all(p['audits']['hashes_unchanged'] for p in parts),fits_unchanged=all(p['audits']['fits_unchanged'] for p in parts),orthonormal_error=max(p['audits']['orthonormal_error'] for p in parts),cached_activation_relative=max(p['audits']['cached_activation_relative'] for p in parts),identity_logits_max_abs=max(p['audits']['identity_logits_max_abs'] for p in parts),previous_layer_screen_margin_replay_max_abs=replay,whole_model_forwards=sum(p['audits']['whole_model_forwards'] for p in parts),test_evaluations=0,finite=all(p['audits']['finite'] for p in parts));assert replay<1e-3;cfg=dict(experiment='H908',layers=[12,13,14],domain='language',scope='Shared bilinearpartner means with full lexical changes; own/opposite/taskpooled/allpooled and linear/quadratic controls. L13/14 screeningselected; L12reference; reusedpanel, no freshconfirmation or compressionclaim.',layer_configs=[p['config'] for p in parts]);(ROOT/f'configs/{BASE}.json').write_text(json.dumps(cfg,indent=2)+'\n');(ART/f'{BASE}.json').write_text(json.dumps(dict(config=cfg,records=records,audits=audits),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
