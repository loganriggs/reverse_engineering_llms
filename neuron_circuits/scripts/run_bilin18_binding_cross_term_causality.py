"""H926 distinguish local routing/value cross term from downstream nonadditivity."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
S='bilin18_binding_cross_term_causality';HEADS=[(14,4),(14,6),(16,3)];MODES=['identity','routing','values','cross','routing_values','full','all_terms']
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'panel':ART/'bilin18_binding_repetition_role_screen_captures.pt','capture':ART/'bilin18_binding_head_factorization_captures.pt','prior':ART/'bilin18_binding_head_factorization.json','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H926','hashes':hashes,'heads':HEADS,'modes':MODES,'design':'All128 existing transfer rows and both contrasts. Exploratory heads selected by H925. Route R=(B-A)V, value U=A(W-V), cross C=(B-A)(W-V). Test R,U,C,R+U,R+U+C, full cached donor-minus-recipient and identity. Native baseline and live downstream. Separate nonadditivity of downstream response from missing local cross term; no new fitting.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['panel'],weights_only=False);cache=torch.load(paths['capture'],weights_only=False);prior=json.loads(paths['prior'].read_text());prev={(r['index'],r['contrast'],r['layer'],r['head'],r['mode']):r for r in prior['records']};ex=data['examples'];ix=[i for i,e in enumerate(ex) if e['split']=='transfer'];keys=['pair','list_order','sender','extra','voice','query'];lookup={tuple(e[k] for k in keys):i for i,e in enumerate(ex)};deltas={};algebra=0.
 for i in ix:
  for c in ['event_swap','same_fact_reword']:
   e=dict(ex[i])
   if c=='event_swap':e['sender']=1-e['sender']
   else:e['voice']='sent_to' if e['voice']=='received_from' else 'received_from'
   j=lookup[tuple(e[k] for k in keys)]
   for l,h in HEADS:
    A=cache[i,l,'A'][h];B=cache[j,l,'A'][h];V=cache[i,l,'V'][:,h];W=cache[j,l,'V'][:,h];R=((B-A)[:,None]*V).sum(0);U=(A[:,None]*(W-V)).sum(0);C=((B-A)[:,None]*(W-V)).sum(0);exact=(B[:,None]*W-A[:,None]*V).sum(0);algebra=max(algebra,float((exact-R-U-C).abs().max()))
    dd={'identity':torch.zeros_like(R),'routing':R,'values':U,'cross':C,'routing_values':R+U,'full':cache[j,l,'H'][h]-cache[i,l,'H'][h],'all_terms':R+U+C}
    for mode,v in dd.items():deltas[i,c,l,h,mode]=v
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 records=[];actual=0;identity=0.;replay=0.;fullerr=0.;priorerr=0.
 for st in range(0,len(ix),16):
  batch=ix[st:st+16];tokens=torch.tensor([ex[i]['ids'] for i in batch],device='cuda');native=final_logits(model,tokens).double().cpu();actual+=len(batch)
  for n,i in enumerate(batch):replay=max(replay,float((native[n]-data['native'][i]).abs().max()))
  for l,h in HEADS:
   for c in ['event_swap','same_fact_reword']:
    full=None
    for mode in MODES:
     delta=torch.stack([deltas[i,c,l,h,mode] for i in batch]).cuda()
     def edit(mod,args):
      x=args[0].clone();x[:,-1,h*128:(h+1)*128]+=delta.to(x);return (x,)
     handle=model.blocks[l].attn.c_proj.register_forward_pre_hook(edit)
     try:y=final_logits(model,tokens).double().cpu()
     finally:handle.remove()
     actual+=len(batch)
     if mode=='identity':identity=max(identity,float((native-y).abs().max()))
     if mode=='full':full=y.clone()
     if mode=='all_terms':fullerr=max(fullerr,float((full-y).abs().max()))
     for n,i in enumerate(batch):
      e=ex[i];second=1-e['sender'] if e['voice']=='sent_to' else e['sender'];v=e['answer_ids'][second];u=e['answer_ids'][1-second];drop=float(native[n,v]-native[n,u]-y[n,v]+y[n,u]);lp=native[n].log_softmax(-1);lq=y[n].log_softmax(-1)
      if mode in ['routing','values','full']:priorerr=max(priorerr,abs(drop-prev[i,c,l,h,'mixed_values' if mode=='values' else mode]['margin_drop']))
      records.append({k:e[k] for k in ['pair','voice','query','sender','extra','list_order']}|{'index':i,'contrast':c,'layer':l,'head':h,'mode':mode,'drop':drop,'KL':float((lp.exp()*(lp-lq)).sum()),'second_preferred':bool(y[n,v]>y[n,u])})
  print('DONE',st,actual,flush=True)
 d={(r['index'],r['contrast'],r['layer'],r['head'],r['mode']):r for r in records};summary=[]
 for c in ['event_swap','same_fact_reword']:
  for l,h in HEADS:
   arr=lambda m:np.array([d[i,c,l,h,m]['drop'] for i in ix]);R,U,C,RU,F=[arr(m) for m in ['routing','values','cross','routing_values','full']];rms=lambda v:float(np.mean(v*v)**.5)
   summary.append(dict(contrast=c,layer=l,head=h,full_RMS=rms(F),naive_sum_error_RMS=rms(F-R-U),downstream_RU_nonadditivity_RMS=rms(RU-R-U),cross_increment_RMS=rms(F-RU),cross_alone_RMS=rms(C),cross_context_nonadditivity_RMS=rms(F-RU-C),all_terms_vs_full_RMS=rms(arr('all_terms')-F),cross_mean=float(C.mean()),cross_increment_mean=float((F-RU).mean())))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_max_abs':identity,'native_replay_max_abs':replay,'all_terms_vs_full_logits_max_abs':fullerr,'prior_effect_replay_max_abs':priorerr,'algebra_max_abs':algebra,'whole_model_forwards':actual,'expected_forwards':128*(1+3*2*len(MODES)),'finite':all(np.isfinite(r['drop']) and np.isfinite(r['KL']) for r in records),'test_evaluations':0}
 assert audits['hashes_unchanged'] and identity==0 and replay<1e-4 and fullerr<1e-4 and priorerr<1e-4 and algebra<1e-10 and actual==audits['expected_forwards'] and audits['finite'],audits
 (ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(summary,audits,flush=True)
if __name__=='__main__':main()
