"""H919 discovery-only paired value spans and causal transfer."""
from run_bilin18_binding_route_value_factorial import ROOT,ART,sha,SNAPSHOT,load_full_model,final_logits,HEADS
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
S='bilin18_binding_value_subspace_transfer'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'runner':Path(__file__),'panel':ART/'bilin18_multidomain_matched_competence_captures.pt','prior':ART/'bilin18_binding_route_value_factorial_logits.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()}
 modes=['identity','full']+[f'paired_{r}' for r in [1,2,4]]+[f'random{s}_{r}' for s in [0,1,2] for r in [1,2,4]]
 cfg={'experiment':'H919','hashes':hashes,'heads':HEADS,'modes':modes,'design':'Fit uncentered SVD on four discovery pair bus differences, one copyA endpoint per pair; no transfer fitting. Same subspace across task/copy and templates. Ranks1/2/4 and three seeded nested random controls per rank. Project bus difference then retain native recipient routing. Compare with full bus edit, and doing nothing. Transfer names previously used, not pristine holdout.'}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['panel'],weights_only=False);prior=torch.load(paths['prior'],weights_only=False);ex=data['examples'];allix=[i for i,e in enumerate(ex) if e['domain']=='binding'];pairs=sorted({ex[i]['pair'] for i in allix});lookup={(ex[i]['id'],ex[i]['order']):i for i in allix};ix=[i for i in allix if ex[i]['pair'] in pairs[4:]];assert len(ix)==32
 fits={};metadata=[];orth=0.;spanerr=0.;staticerr=0.
 for l,h in HEADS:
  fitids=[i for i in allix if ex[i]['pair'] in pairs[:4] and ex[i]['family']=='copy' and ex[i]['format']=='A' and ex[i]['order']==0];assert len(fitids)==4
  X=torch.stack([(data['bus'][lookup[ex[i]['id'],1]][ex[i]['source_positions'][0],h]-data['bus'][i][ex[i]['source_positions'][0],h]).double() for i in fitids]);_,sing,V=torch.linalg.svd(X,full_matrices=False)
  spanerr=max(spanerr,float((X-X@V.T@V).norm()/X.norm()))
  for r in [1,2,4]:fits[l,h,f'paired_{r}']=V[:r]
  for s in [0,1,2]:
   gen=torch.Generator().manual_seed(91900+10*h+s);Q=torch.linalg.qr(torch.randn(128,4,generator=gen,dtype=torch.double),mode='reduced').Q.T
   for r in [1,2,4]:fits[l,h,f'random{s}_{r}']=Q[:r]
  metadata.append({'layer':l,'head':h,'fit_indices':fitids,'singular_values':sing.tolist()})
 for B in fits.values():orth=max(orth,float((B@B.T-torch.eye(len(B))).abs().max()))
 # Verify this static bus depends on token identity, not task/template context.
 seen={}
 for i in allix:
  for p in ex[i]['source_positions']:
   for l,h in HEADS:
    key=(ex[i]['ids'][p],h);v=data['bus'][i][p,h]
    if key in seen:staticerr=max(staticerr,float((v-seen[key]).abs().max()))
    else:seen[key]=v
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 deltas={};energy={}
 for i in ix:
  j=lookup[ex[i]['id'],1-ex[i]['order']]
  for l,h in HEADS:
   db=(data['bus'][j][:,h]-data['bus'][i][:,h]).double();A=data['captures'][i][l,h]['routing'].double();coef=float(model.blocks[l].attn.lamb)
   for mode in modes:
    if mode=='identity':proj=torch.zeros_like(db)
    elif mode=='full':proj=db
    else:B=fits[l,h,mode];proj=db@B.T@B
    energy[i,l,h,mode]=float(proj.square().sum()/db.square().sum());deltas[i,l,h,mode]=coef*(A[:,None]*proj).sum(0)
 buckets=defaultdict(list)
 for i in ix:buckets[len(ex[i]['ids'])].append(i)
 records=[];actual=0;replay=0.;identity=0.;fullreplay=0.
 for length,inds in sorted(buckets.items()):
  for st in range(0,len(inds),8):
   batch=inds[st:st+8];tokens=torch.tensor([ex[i]['ids'] for i in batch],device='cuda');native=final_logits(model,tokens).double().cpu();actual+=len(batch)
   for n,i in enumerate(batch):replay=max(replay,float((native[n]-data['native'][i]).abs().max()))
   for l,h in HEADS:
    for mode in modes:
     delta=torch.stack([deltas[i,l,h,mode] for i in batch]).cuda()
     def hook(mod,args):
      x=args[0].clone();x[:,-1,h*128:(h+1)*128]+=delta.to(x);return (x,)
     handle=model.blocks[l].attn.c_proj.register_forward_pre_hook(hook)
     try:y=final_logits(model,tokens).double().cpu()
     finally:handle.remove()
     actual+=len(batch)
     for n,i in enumerate(batch):
      e=ex[i];exact=prior[i,l,h,'bus'];ids=[g[0] for g in e['groups']];o=e['expected'];margin=lambda z:float(z[ids[o]]-z[ids[1-o]])
      if mode=='identity':identity=max(identity,float((native[n]-y[n]).abs().max()))
      if mode=='full':fullreplay=max(fullreplay,float((exact-y[n]).abs().max()))
      lp=exact.log_softmax(-1);lq=y[n].log_softmax(-1);ln=native[n].log_softmax(-1)
      records.append({k:e[k] for k in ['pair','family','format','order']}|{'layer':l,'head':h,'mode':mode,'energy_fraction':energy[i,l,h,mode],'exact_effect':margin(exact)-margin(native[n]),'approx_effect':margin(y[n])-margin(native[n]),'effect_error':margin(y[n])-margin(exact),'exact_to_approx_KL':float((lp.exp()*(lp-lq)).sum()),'exact_to_native_KL':float((lp.exp()*(lp-ln)).sum())})
   print('DONE',length,st,actual,flush=True)
 groups=defaultdict(list)
 for r in records:groups[r['family'],r['layer'],r['head'],r['mode']].append(r)
 summary=[]
 for key,rr in groups.items():
  rms=lambda k:float(np.mean([r[k]**2 for r in rr])**.5)
  baseline=float(np.mean([r['exact_to_native_KL'] for r in rr]));kl=float(np.mean([r['exact_to_approx_KL'] for r in rr]))
  summary.append(dict(zip(['family','layer','head','mode'],key))|{'energy':float(np.mean([r['energy_fraction'] for r in rr])),'effect_error_ratio':rms('effect_error')/rms('exact_effect'),'KL':kl,'KL_ratio':kl/baseline})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_replay_max_abs':replay,'identity_max_abs':identity,'full_bus_replay_max_abs':fullreplay,'basis_orthogonality':orth,'discovery_rank4_reconstruction_relative':spanerr,'static_bus_token_max_abs':staticerr,'whole_model_forwards':actual,'expected_forwards':32*(1+3*len(modes)),'test_evaluations':0,'finite':all(np.isfinite(r['KL_ratio']) and np.isfinite(r['effect_error_ratio']) for r in summary)}
 assert audits['hashes_unchanged'] and replay<1e-4 and identity==0 and fullreplay<1e-4 and orth<1e-10 and spanerr<1e-10 and staticerr<1e-4 and actual==audits['expected_forwards'] and audits['finite'],audits
 torch.save({'fits':fits,'metadata':metadata},ART/f'{S}_fits.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'fits':metadata,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
