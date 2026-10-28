"""H928 token aggregation audit and full-distribution aligned-routing fidelity."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
S='bilin18_identity_aggregation_fidelity';HEADS=[(14,4),(14,6),(16,3)]
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'panel':ART/'bilin18_binding_repetition_role_screen_captures.pt','capture':ART/'bilin18_binding_head_factorization_captures.pt','broad':ART/'bilin18_multidomain_matched_competence_captures.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H928','hashes':hashes,'heads':HEADS,'design':'A: audit exact persistent-value token aggregation, with position provenance, across existing agreement/language/binding/copy at previously captured heads. No training, native-output proof only for local component. B: compare full distribution of full donor head swap against identity-aligned routing and position-aligned routing on128 QA transfer rows, both contrasts, all3 exploratory heads. No outcome selection.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 broad=torch.load(paths['broad'],weights_only=False);dictionary={};agg=[];provenance=[];staticerr=0.;aggerr=0.;relerr=0.
 for i,e in enumerate(broad['examples']):
  if e['domain'] not in ['agreement','language','binding']:continue
  groups=defaultdict(list)
  for p,t in enumerate(e['ids']):groups[t].append(p)
  provenance.append({'index':i,'groups':dict(groups)})
  for l,h in [(8,3),(13,0),(14,4)]:
   A=broad['captures'][i][l,h]['routing'].double();V=broad['bus'][i][:,h].double();native=(A[:,None]*V).sum(0);grouped=torch.zeros_like(native)
   for token,ps in groups.items():
    key=token,h
    if key not in dictionary:dictionary[key]=V[ps[0]].clone()
    v=dictionary[key];staticerr=max(staticerr,float((V[ps]-v).abs().max()));grouped+=A[ps].sum()*v
   err=float((grouped-native).norm());rel=err/float(native.norm().clamp_min(1e-12));aggerr=max(aggerr,float((grouped-native).abs().max()));relerr=max(relerr,rel);agg.append(dict(index=i,domain=e['domain'],family=e['family'],layer=l,head=h,positions=len(e['ids']),identities=len(groups),relative_error=rel,absolute_norm_error=err))
 assert len(agg)==576,len(agg)
 data=torch.load(paths['panel'],weights_only=False);cache=torch.load(paths['capture'],weights_only=False);ex=data['examples'];ix=[i for i,e in enumerate(ex) if e['split']=='transfer'];keys=['pair','list_order','sender','extra','voice','query'];lookup={tuple(e[k] for k in keys):i for i,e in enumerate(ex)};deltas={};invariance=0.
 for i in ix:
  for c in ['event_swap','same_fact_reword']:
   e=dict(ex[i])
   if c=='event_swap':e['sender']=1-e['sender']
   else:e['voice']='sent_to' if e['voice']=='received_from' else 'received_from'
   j=lookup[tuple(e[k] for k in keys)]
   for l,h in HEADS:
    A=cache[i,l,'A'][h];B=cache[j,l,'A'][h];V=cache[i,l,'V'][:,h];W=cache[j,l,'V'][:,h];perm=torch.arange(len(A));p,q=ex[i]['roles']['event_sender'],ex[i]['roles']['event_recipient'];perm[p]=q;perm[q]=p;assert ex[i]['ids'][p]==ex[j]['ids'][int(perm[p])]
    invariance=max(invariance,float(((B[:,None]*W).sum(0)-(B[perm,None]*W[perm]).sum(0)).abs().max()))
    for mode,delta in [('full',cache[j,l,'H'][h]-cache[i,l,'H'][h]),('aligned_routing',((B[perm]-A)[:,None]*V).sum(0)),('position_routing',((B-A)[:,None]*V).sum(0))]:deltas[i,c,l,h,mode]=delta
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 records=[];actual=0;replay=0.
 for st in range(0,len(ix),16):
  batch=ix[st:st+16];tokens=torch.tensor([ex[i]['ids'] for i in batch],device='cuda');native=final_logits(model,tokens).double().cpu();actual+=len(batch)
  for n,i in enumerate(batch):replay=max(replay,float((native[n]-data['native'][i]).abs().max()))
  for l,h in HEADS:
   for c in ['event_swap','same_fact_reword']:
    full=None
    for mode in ['full','aligned_routing','position_routing']:
     delta=torch.stack([deltas[i,c,l,h,mode] for i in batch]).cuda()
     def edit(mod,args):
      x=args[0].clone();x[:,-1,h*128:(h+1)*128]+=delta.to(x);return (x,)
     handle=model.blocks[l].attn.c_proj.register_forward_pre_hook(edit)
     try:y=final_logits(model,tokens).double().cpu()
     finally:handle.remove()
     actual+=len(batch)
     if mode=='full':full=y.clone()
     for n,i in enumerate(batch):
      lf=full[n].log_softmax(-1);la=y[n].log_softmax(-1);ln=native[n].log_softmax(-1);records.append(dict(index=i,pair=ex[i]['pair'],voice=ex[i]['voice'],query=ex[i]['query'],contrast=c,layer=l,head=h,mode=mode,full_to_approx_KL=float((lf.exp()*(lf-la)).sum()),full_to_native_KL=float((lf.exp()*(lf-ln)).sum()),max_logit_error=float((full[n]-y[n]).abs().max()),top1_matches_full=int(full[n].argmax())==int(y[n].argmax())))
  print('DONE',st,actual,flush=True)
 groups=defaultdict(list)
 for r in records:groups[r['contrast'],r['layer'],r['head'],r['mode']].append(r)
 summary=[]
 for key,rr in groups.items():
  kl=float(np.mean([r['full_to_approx_KL'] for r in rr]));base=float(np.mean([r['full_to_native_KL'] for r in rr]));summary.append(dict(zip(['contrast','layer','head','mode'],key))|{'KL':kl,'baseline_KL':base,'KL_ratio':kl/base,'max_logit_error':max(r['max_logit_error'] for r in rr),'top1_match':float(np.mean([r['top1_matches_full'] for r in rr]))})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_replay_max_abs':replay,'aggregation_max_abs':aggerr,'aggregation_max_relative_norm':relerr,'same_token_value_max_abs':staticerr,'permutation_invariance_max_abs':invariance,'aggregation_rows':len(agg),'whole_model_forwards':actual,'expected_forwards':128*(1+3*2*3),'finite':all(np.isfinite(r['KL_ratio']) for r in summary),'test_evaluations':0}
 assert audits['hashes_unchanged'] and replay<1e-4 and staticerr<1e-4 and aggerr<1e-4 and relerr<1e-4 and invariance<1e-10 and actual==audits['expected_forwards'] and audits['finite'],audits
 torch.save(dictionary,ART/f'{S}_dictionary.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'aggregation':agg,'position_provenance':provenance,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(summary,audits,flush=True)
if __name__=='__main__':main()
