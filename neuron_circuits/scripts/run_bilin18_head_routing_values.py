"""H738: native signed routing/value factorial at fixed heads."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch,numpy as np
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_head_routing_values'
def factor_deltas(a,b,Lr,Ld,Br,Bd,lam):
 Vr=(1-lam)*Lr+lam*Br;Vd=(1-lam)*Ld+lam*Bd
 return dict(full=b@Vd-a@Vr,routing=(b-a)@Vr,values=a@(Vd-Vr),local=(1-lam)*(a@(Ld-Lr)),bus=lam*(a@(Bd-Br)),interaction=(b-a)@(Vd-Vr),identity=torch.zeros_like(a@Vr))
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'fresh':ART/'bilin18_fresh_head_tasks_captures.pt','fresh_results':ART/'bilin18_fresh_head_tasks.json','manifest':ART/'bilin18_fresh_head_tasks_manifest.json','model':SNAPSHOT/'pytorch_model.bin','model_source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py'),'runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py'};hashes={k:sha(p) for k,p in paths.items()};old=torch.load(paths['fresh'],weights_only=False);prior=json.loads(paths['fresh_results'].read_text());rows=json.loads(paths['manifest'].read_text());examples=old['examples'];sites={'agreement_head':(11,3),'capital_head':(13,0)};excluded=[];valid=[]
 for row in rows:
  if (row['id'],0) not in examples:continue
  for rv,dv in [(0,1),(1,0),(0,2)]:
   er,ed=examples[row['id'],rv],examples[row['id'],dv]
   if len(er['ids'])!=len(ed['ids']):excluded.append(dict(id=row['id'],receiver=rv,donor=dv,reason='unequal token length',lengths=[len(er['ids']),len(ed['ids'])]));continue
   valid.append(dict(id=row['id'],family=row['family'],split=row['split'],receiver=rv,donor=dv,changed_positions=[i for i,(a,b) in enumerate(zip(er['ids'],ed['ids'])) if a!=b]))
 cfg=dict(experiment='H738',hashes=hashes,sites=sites,design='FreezeH732headsites andpanel. Capture native last-querysignedpattern A=(q.k/D)*(q2.k2/D) withRMSnorm/RoPE, locallayerV andpersistentlayer0busB. H=A[(1-lambda)L+lambdaB]. Equal-length recipient/donor examples alignedbyabsoluteposition only; structuralexclusions lockedbeforeoutputs. Add routingonly, valuesonly, localonly,busonly, bilinearinteractiononly, full andzeroidentity deltas torecipienthead; downstreamlive. Fullvector delta=route+value+interaction; value=local+bus. No learnedfeaturizer/selection.',limitations='Mixing internalstates is a controlled counterfactual, notproofnaturalalgorithm. Routing andlocalvalues alreadycontextual. Equal-lengthpositionmatching notsemanticrealignment. Failedcompetence strata diagnostic; nuisance availabilityreported separately. Signedpatterns andlambda arenotprobabilities orconvexweights. Modeloutput effects neednotadd; these formulas are exactheadvector identities.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda');assert model.config.squared_attn
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 native=old['native_logits'];cache={};reconstruction=[];native_replay=[];forwards=0;buckets=defaultdict(list)
 for k,e in examples.items():buckets[len(e['ids'])].append(k)
 lambdas={site:float(model.blocks[layer].attn.lamb) for site,(layer,head) in sites.items()}
 for length,keys in buckets.items():
  for lo in range(0,len(keys),12):
   bb=keys[lo:lo+12];cap={};hooks=[]
   for site,(layer,head) in sites.items():
    attn=model.blocks[layer].attn
    def pre(m,args,site=site,head=head):cap[site,'bus']=args[1].view(len(bb),length,9,128)[:,:,head].cpu()
    hooks.append(attn.register_forward_pre_hook(pre))
    for name in ['c_q','c_k','c_q2','c_k2','c_v']:
     def h(m,args,out,site=site,head=head,name=name):cap[site,name]=out[:,:,head*128:(head+1)*128].cpu()
     hooks.append(getattr(attn,name).register_forward_hook(h))
    def headpre(m,args,site=site,head=head):cap[site,'head']=args[0][:,-1,head*128:(head+1)*128].cpu()
    hooks.append(attn.c_proj.register_forward_pre_hook(headpre))
   try:logits,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:
    for h in hooks:h.remove()
   forwards+=len(bb)
   for n,k in enumerate(bb):native_replay.append(float((logits[n].cpu()-native[k]).abs().max()))
   for site,(layer,head) in sites.items():
    attn=model.blocks[layer].attn;q,k,q2,k2=[cap[site,n].cuda()[:,:,None,:] for n in ['c_q','c_k','c_q2','c_k2']];cos,sin=attn.rotary(q);q,k,q2,k2=[model.tt.apply_rotary_emb(F.rms_norm(t,(128,)),cos,sin) for t in [q,k,q2,k2]];a=((q[:,-1]*k[:,:,0]).sum(-1)/128)*((q2[:,-1]*k2[:,:,0]).sum(-1)/128);L=cap[site,'c_v'].cuda();B=cap[site,'bus'].cuda();lam=lambdas[site];v=(1-lam)*L+lam*B;headrec=torch.einsum('bt,btd->bd',a,v);true=cap[site,'head'].cuda();reconstruction.extend(((headrec-true).norm(dim=-1)/true.norm(dim=-1).clamp_min(1e-12)).tolist())
    for n,key in enumerate(bb):cache[site,key[0],key[1]]=dict(pattern=a[n].cpu(),local=L[n].cpu(),bus=B[n].cpu(),head=true[n].cpu())
 assert max(reconstruction)<1e-4 and max(native_replay)<1e-4
 vectors={};vector_errors=[];value_errors=[]
 for site in sites:
  for r in valid:
   x=cache[site,r['id'],r['receiver']];y=cache[site,r['id'],r['donor']];d=factor_deltas(x['pattern'].double(),y['pattern'].double(),x['local'].double(),y['local'].double(),x['bus'].double(),y['bus'].double(),lambdas[site]);scale=d['full'].norm().clamp_min(1e-12);vector_errors.append(float((d['full']-d['routing']-d['values']-d['interaction']).norm()/scale));value_errors.append(float((d['values']-d['local']-d['bus']).norm()/scale));vectors[site,r['id'],r['receiver'],r['donor']]=d
 previous={(r['site'],r['id'],r['receiver'],r['donor']):r for r in prior['records']};records=[];full_replay=[]
 for site,(layer,head) in sites.items():
  for mode in ['full','routing','values','local','bus','interaction','identity']:
   jobs=defaultdict(list)
   for r in valid:jobs[len(examples[r['id'],r['receiver']]['ids'])].append(r)
   for length,jj in jobs.items():
    for lo in range(0,len(jj),12):
     bb=jj[lo:lo+12];delta=torch.stack([vectors[site,r['id'],r['receiver'],r['donor']][mode] for r in bb]).cuda().float()
     def edit(m,args):
      y=args[0].clone();y[:,-1,head*128:(head+1)*128]+=delta.to(y);return (y,)
     hook=model.blocks[layer].attn.c_proj.register_forward_pre_hook(edit)
     try:logits,_=run(model,torch.tensor([examples[r['id'],r['receiver']]['ids'] for r in bb],device='cuda'))
     finally:hook.remove()
     forwards+=len(bb)
     for n,r in enumerate(bb):
      key=(r['id'],r['receiver']);gs=examples[key]['groups'];base=measure(native[key],gs);ed=measure(logits[n],gs);change=ed['margin']-base['margin'];sign=1 if r['donor'] in [0,2] else -1;records.append(dict(**r,site=site,mode=mode,margin_change=change,donor_directed_margin_change=sign*change,edited_donor_correct=ed['top1'][0 if r['donor']==2 else r['donor']]))
      if mode=='full':full_replay.append(abs(change-previous[site,r['id'],r['receiver'],r['donor']]['margin_change']))
   print('DONE',site,mode,flush=True)
 identity=max(abs(r['margin_change']) for r in records if r['mode']=='identity');audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['margin_change']) for r in records),native_logits_replay_max_abs=max(native_replay),native_head_factorization_max_relative=max(reconstruction),full_delta_identity_max_relative=max(vector_errors),values_local_bus_identity_max_relative=max(value_errors),full_head_swap_replay_max_abs=max(full_replay),zero_identity_margin_max_abs=identity,whole_model_forwards=forwards,test_evaluations=0);print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(vector_errors)<1e-8 and max(value_errors)<1e-8 and max(full_replay)<1e-4 and identity<1e-4;torch.save(dict(components=cache,deltas=vectors),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,lambdas=lambdas,excluded=excluded,valid_directions=valid,competence=prior['summary'],records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2))
if __name__=='__main__':main()
