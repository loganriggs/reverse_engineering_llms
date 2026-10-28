"""H905 paired-change versus pooled PCA; finite L12 MLP interventions."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict
import json,time,torch,numpy as np
BASE='bilin18_language_late_subspaces'
@torch.inference_mode()
def run_layer(layer):
 S=BASE+'_l'+str(layer)
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','panel':ART/'bilin18_multidomain_matched_competence_captures.pt','mapping':ART/'bilin18_partner_subspace_fidelity.json','states':ART/'bilin18_partner_subspace_fidelity_native_mlp_states.pt'};hashes={k:sha(p) for k,p in paths.items()};cfg=dict(experiment='H907',hashes=hashes,layer=layer,ranks=[1,2,4],design='Task discovery first4pairs per domain/template. Pooled centered activationPCA vs uncentered SVD of four paired differences (sign invariant), random4. Last4pairs transfer task/copy. Matching-template fits. Finite normalized-MLP-input projected donor changes, without renormalization; downstreamlive, restrecipient. Full change is MLP-only donor substitution, not wholeprompt swap. Existingreused panels, no semanticlabel assigned to endpointorder.');(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');panel=torch.load(paths['panel'],weights_only=False);mapping=json.loads(paths['mapping'].read_text());ex=[panel['examples'][i] for i in mapping['selection_original_indices']];states=torch.load(paths['states'],weights_only=False);raw=torch.stack([states[i,(layer,'raw')] for i in range(len(ex))]).double().cuda();uall=raw/(raw.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();matched={(e['domain'],e['family'],e['format'],e['pair'],e['order']):i for i,e in enumerate(ex)};fits={};evalix=[];meta=[];orth=0.;gen=torch.Generator(device='cuda').manual_seed(905)
 for dom in ['language']:
  pairs=sorted({e['pair'] for e in ex if e['domain']==dom});evalix.extend(i for i,e in enumerate(ex) if e['domain']==dom and e['pair'] in pairs[4:])
  for fmt in ['A','B']:
   ix=[i for i,e in enumerate(ex) if e['domain']==dom and e['family']=='task' and e['format']==fmt and e['pair'] in pairs[:4]];X=uall[ix];deltas=torch.stack([uall[matched[dom,'task',fmt,p,1]]-uall[matched[dom,'task',fmt,p,0]] for p in pairs[:4]]);_,_,V=torch.linalg.svd(X-X.mean(0),full_matrices=False);_,s,W=torch.linalg.svd(deltas,full_matrices=False);random=torch.linalg.qr(torch.randn(1152,4,device='cuda',dtype=torch.double,generator=gen)).Q.T;fits[dom,fmt]=dict(pooled=V[:4],paired=W[:4],random=random);orth=max(orth,*[float((B@B.T-torch.eye(4,device='cuda')).abs().max()) for B in [V[:4],W[:4],random]]);meta.append(dict(domain=dom,format=fmt,discovery_indices=ix,paired_singular_values=s.cpu().tolist()))
 assert len(evalix)==32;torch.save({k:{n:v.cpu() for n,v in f.items()} for k,f in fits.items()},ART/f'{S}_fits.pt');fit_hash=sha(ART/f'{S}_fits.pt');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 mod=model.blocks[layer].mlp;L,R,D=[m.weight.double() for m in [mod.Left,mod.Right,mod.Down]];state={'mode':'baseline'};cacheerr=0.;stash={}
 def hook(mod,args,out):
  nonlocal cacheerr
  if state['mode']=='baseline':return
  u=args[0][:,-1].double();ix=state['indices'];donors=[matched[ex[i]['domain'],ex[i]['family'],ex[i]['format'],ex[i]['pair'],1-ex[i]['order']] for i in ix];delta=uall[donors]-u;cacheerr=max(cacheerr,float((u-uall[ix]).norm()/u.norm()));mode=state['mode']
  if mode=='full':v=delta
  elif mode=='identity':v=torch.zeros_like(delta)
  else:
   kind,rank=mode.split('_');B=torch.stack([fits[ex[i]['domain'],ex[i]['format']][kind][:int(rank)] for i in ix]);v=torch.einsum('bk,bkj->bj',torch.einsum('bj,bkj->bk',delta,B),B)
  # Exact bilinear finite change including the quadratic term.
  change=((v@L.T)*(u@R.T)+(u@L.T)*(v@R.T)+(v@L.T)*(v@R.T))@D.T;stash['capture']=((v.square().sum(-1))/(delta.square().sum(-1))).cpu();stash['radius_ratio']=((u+v).norm(dim=-1)/u.norm(dim=-1)).cpu();z=out.clone();z[:,-1]+=change.to(out);return z
 handle=mod.register_forward_hook(hook);modes=['identity','full','pooled_1','pooled_2','pooled_4','paired_1','paired_2','paired_4','random_4'];buckets=defaultdict(list)
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
      r=dict(index=i,domain=e['domain'],family=e['family'],format=e['format'],pair=e['pair'],mode=mode,KL=float((lp.exp()*(lp-lq)).sum()),native_margin=sign*float(base[n,ids[0]]-base[n,ids[1]]),margin=sign*float(y[n,ids[0]]-y[n,ids[1]]),native_top1_correct=int(base[n].argmax())==ids[e['expected']],capture=float(stash['capture'][n]),radius_ratio=float(stash['radius_ratio'][n]))
      if mode!='identity':ref=refs[i];r['full_to_approx_KL']=float((ref.exp()*(ref-lq)).sum())
      records.append(r)
   print('DONE',length,actual,flush=True)
 finally:handle.remove()
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),fits_unchanged=sha(ART/f'{S}_fits.pt')==fit_hash,orthonormal_error=orth,cached_activation_relative=cacheerr,identity_logits_max_abs=identity,whole_model_forwards=actual,expected_forwards=320,test_evaluations=0,finite=all(np.isfinite(r['KL']) for r in records));assert audits['hashes_unchanged'] and audits['fits_unchanged'] and orth<1e-10 and cacheerr<1e-5 and identity<1e-4 and actual==320 and audits['finite'];(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,fitmeta=meta,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
def main():
 for layer in [12,13,14]:run_layer(layer)
 parts=[json.loads((ART/f'{BASE}_l{k}.json').read_text()) for k in [12,13,14]];records=[dict(r,layer=k) for k,p in zip([12,13,14],parts) for r in p['records']];prior=json.loads((ART/'bilin18_multidomain_mlp_localization.json').read_text());lookup={(r['index'],r['mode']):r for r in prior['records']};replay=max(abs(r['margin']-lookup[r['index'],'layer_'+str(r['layer'])]['margin']) for r in records if r['mode']=='full');audits=dict(hashes_unchanged=all(p['audits']['hashes_unchanged'] for p in parts),fits_unchanged=all(p['audits']['fits_unchanged'] for p in parts),orthonormal_error=max(p['audits']['orthonormal_error'] for p in parts),cached_activation_relative=max(p['audits']['cached_activation_relative'] for p in parts),identity_logits_max_abs=max(p['audits']['identity_logits_max_abs'] for p in parts),previous_layer_screen_margin_replay_max_abs=replay,whole_model_forwards=sum(p['audits']['whole_model_forwards'] for p in parts),test_evaluations=0,finite=all(p['audits']['finite'] for p in parts));assert replay<1e-3;cfg=dict(experiment='H907',layers=[12,13,14],domain='language',scope='L13/14 selected from H906 screening, L12 retained. Reused evaluation panel, discovery fits disjointpairs. No freshconfirmation claim.',layer_configs=[p['config'] for p in parts]);(ROOT/f'configs/{BASE}.json').write_text(json.dumps(cfg,indent=2)+'\n');(ART/f'{BASE}.json').write_text(json.dumps(dict(config=cfg,records=records,audits=audits),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
