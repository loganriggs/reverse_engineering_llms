"""H933 disjoint lexical-fit stability with common evaluation."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time,sys
from run_bilin18_query_prototype_screen import kmeans
FIT_TEMPLATE=sys.argv[1] if len(sys.argv)>1 else 'A';assert FIT_TEMPLATE in ['A','B']
FIT_SUBSET=int(sys.argv[2]) if len(sys.argv)>2 else 0;assert FIT_SUBSET in [0,1]
S='bilin18_disjoint_query_prototype_stability_'+FIT_TEMPLATE+str(FIT_SUBSET);HEADS=[(8,3),(13,0),(14,4)];MODES=['native_routing','zero','context_mean','distance_mean','family_distance']+[f'proto{k}s{s}' for k in [1,4,8] for s in [0,1,2]]
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'panel':ART/'bilin18_multidomain_matched_competence_captures.pt','prototype':ART/'bilin18_query_prototype_screen_fits.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H933','fit_subset':FIT_SUBSET,'fit_template':FIT_TEMPLATE,'hashes':hashes,'heads':HEADS,'modes':MODES,'design':'Refit taskblind k1/4/8 query prototypes and routing means using one of two disjoint 2-pair lexical subsets per domain, one template only; same seeds as H932. Evaluate last4 pairs on BOTH templates. All3 seeds, live gates, joint3heads and donor-bus controls. Uncovered cluster distances fall back to pooled training distance; globally absent distances use0. Context baseline reuses fit-template absolute-position means, absent positions0. No evaluation fit or outcome selection.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];allix=[i for i,e in enumerate(ex) if e['domain'] in ['agreement','language','binding']];pairs={d:sorted({ex[i]['pair'] for i in allix if ex[i]['domain']==d}) for d in ['agreement','language','binding']};train=[i for i in allix if ex[i]['pair'] in pairs[ex[i]['domain']][2*FIT_SUBSET:2*FIT_SUBSET+2] and ex[i]['format']==FIT_TEMPLATE];ix=[i for i in allix if ex[i]['pair'] in pairs[ex[i]['domain']][4:]];assert len(train)==24 and len(ix)==96;lookup={(ex[i]['id'],ex[i]['order']):i for i in allix};donors={i:lookup[ex[i]['id'],1-ex[i]['order']] for i in ix};groups=defaultdict(list);fitids=defaultdict(set)
 for i in train:
  e=ex[i]
  for l,h in HEADS:
   A=data['captures'][i][l,h]['routing'].double()
   for p,a in enumerate(A):
    for key in [('context_mean',e['domain'],e['family'],e['format'],l,h,p),('distance_mean',l,h,len(A)-1-p),('family_distance',e['family'],l,h,len(A)-1-p)]:groups[key].append(float(a));fitids[key].add(i)
 fits={k:float(np.mean(v)) for k,v in groups.items()};pred={}
 for i in ix:
  e=ex[i];T=len(e['ids'])
  for l,h in HEADS:
   for mode in MODES[1:]:
    if mode.startswith('proto'):continue
    if mode=='zero':v=torch.zeros(T,dtype=torch.double)
    else:
     kk=[(mode,e['domain'],e['family'],FIT_TEMPLATE,l,h,p) if mode=='context_mean' else (mode,l,h,T-1-p) if mode=='distance_mean' else (mode,e['family'],l,h,T-1-p) for p in range(T)];v=torch.tensor([fits.get(k,0.) for k in kk],dtype=torch.double)
    pred[i,l,h,mode]=v
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 proto=torch.load(paths['prototype'],weights_only=False);protofits={}
 for l,h in HEADS:
  X=torch.stack([proto['queries'][i,l,h] for i in train])
  for k in [1,4,8]:
   for seed in [0,1,2]:
    C,labels,it=kmeans(X,k,9320+seed);bins=defaultdict(list)
    for n,i in enumerate(train):
     aa=data['captures'][i][l,h]['routing']
     for p,a in enumerate(aa):bins[int(labels[n]),len(aa)-1-p].append(float(a))
    protofits[l,h,k,seed]={'centers':C,'coefficients':{key:float(np.mean(v)) for key,v in bins.items()},'train_indices':train,'iterations':it}
 assert all(ex[i]['format']==FIT_TEMPLATE for i in train)
 centers={key:v['centers'].cuda() for key,v in protofits.items()};queries={};gatelog=[];fallback=0
 state={'mode':'native_routing','donor':False,'batch':[]};live={};handles=[];originals={}
 for l,h in HEADS:
  for which in ['q','q2']:
   def cap(mod,args,out,l=l,h=h,which=which):
    z=out[:,-1].reshape(-1,9,128)[:,h];queries[l,h,which]=torch.nn.functional.rms_norm(z,(128,)).double()
   handles.append(getattr(model.blocks[l].attn,'c_'+which).register_forward_hook(cap))
  fn=model.blocks[l].attn.squared_attention;originals[l]=fn
  def capture(q,k,v,q2,k2,l=l,h=h,fn=fn):
   live[l,h]=((torch.einsum('bhd,bthd->bht',q[:,-1],k)/128)*(torch.einsum('bhd,bthd->bht',q2[:,-1],k2)/128))[:,h];return fn(q,k,v,q2,k2)
  model.blocks[l].attn.squared_attention=capture
  def edit(mod,args,l=l,h=h):
   nonlocal fallback
   if state['mode']=='native_routing' and not state['donor']:return
   dd=[]
   for n,i in enumerate(state['batch']):
    A=live[l,h][n].double();mode=state['mode']
    if mode.startswith('proto'):
     k,seed=map(int,mode[5:].split('s'));key=l,h,k,seed;q=torch.cat([queries[l,h,'q'][n],queries[l,h,'q2'][n]]);cluster=int((centers[key]-q).square().sum(-1).argmin());means=protofits[key]['coefficients'];vv=[]
     for p in range(len(A)):
      poskey=cluster,len(A)-1-p
      if poskey not in means:fallback+=1
      vv.append(means.get(poskey,fits.get(('distance_mean',l,h,len(A)-1-p),0.)))
     B=torch.tensor(vv,dtype=torch.double,device='cuda');nativecluster=int((centers[key].cpu()-proto['queries'][i,l,h]).square().sum(-1).argmin());gatelog.append(dict(index=i,format=ex[i]['format'],mode=mode,donor=state['donor'],layer=l,head=h,cluster=cluster,native_cluster=nativecluster,changed=cluster!=nativecluster))
    else:B=A if mode=='native_routing' else pred[i,l,h,mode].cuda()
    V=data['bus'][i][:,h].double().cuda();j=donors[i] if state['donor'] else i;W=data['bus'][j][:,h].double().cuda();dd.append(float(model.blocks[l].attn.lamb)*(B[:,None]*W-A[:,None]*V).sum(0))
   x=args[0].clone();x[:,-1,h*128:(h+1)*128]+=torch.stack(dd).to(x);return (x,)
  handles.append(model.blocks[l].attn.c_proj.register_forward_pre_hook(edit))
 buckets=defaultdict(list)
 for i in ix:buckets[len(ex[i]['ids'])].append(i)
 records=[];actual=0;replay=0.;identity=0.
 try:
  for length,inds in sorted(buckets.items()):
   for st in range(0,len(inds),8):
    batch=inds[st:st+8];state.update(mode='native_routing',donor=False,batch=batch);tokens=torch.tensor([ex[i]['ids'] for i in batch],device='cuda');native=final_logits(model,tokens).double().cpu();actual+=len(batch)
    for n,i in enumerate(batch):replay=max(replay,float((native[n]-data['native'][i]).abs().max()))
    yy={}
    for mode in MODES:
     for donor in [False,True]:
      state.update(mode=mode,donor=donor);yy[mode,donor]=final_logits(model,tokens).double().cpu();actual+=len(batch)
    identity=max(identity,float((native-yy['native_routing',False]).abs().max()))
    for mode in MODES:
     for n,i in enumerate(batch):
      e=ex[i];v=e['groups'][e['expected']][0];u=e['groups'][1-e['expected']][0];margin=lambda z:float(z[v]-z[u]);z=yy[mode,False][n];w=yy[mode,True][n];ref=yy['native_routing',True][n];lp=native[n].log_softmax(-1);lz=z.log_softmax(-1);lf=ref.log_softmax(-1);lw=w.log_softmax(-1)
      records.append({k:e[k] for k in ['domain','family','format','pair','order']}|{'index':i,'mode':mode,'native_KL':float((lp.exp()*(lp-lz)).sum()),'edited_KL':float((lf.exp()*(lf-lw)).sum()),'native_margin_error':margin(z)-margin(native[n]),'reference_effect':margin(ref)-margin(native[n]),'predicted_effect':margin(w)-margin(z),'effect_error':margin(w)-margin(z)-margin(ref)+margin(native[n]),'native_top1':int(native[n].argmax())==v,'predicted_top1':int(z.argmax())==v})
    print('DONE',length,st,actual,flush=True)
 finally:
  for handle in handles:handle.remove()
  for l,fn in originals.items():model.blocks[l].attn.squared_attention=fn
 grouped=defaultdict(list)
 for r in records:grouped[r['domain'],r['family'],r['format'],r['mode']].append(r)
 summary=[]
 for key,rr in grouped.items():
  rms=lambda k:float(np.mean([r[k]**2 for r in rr])**.5);summary.append(dict(zip(['domain','family','format','mode'],key))|{'native_KL':float(np.mean([r['native_KL'] for r in rr])),'edited_KL':float(np.mean([r['edited_KL'] for r in rr])),'native_margin_RMSE':rms('native_margin_error'),'reference_effect_RMS':rms('reference_effect'),'effect_RMSE':rms('effect_error'),'effect_error_ratio':rms('effect_error')/max(rms('reference_effect'),1e-12),'native_top1':float(np.mean([r['native_top1'] for r in rr])),'predicted_top1':float(np.mean([r['predicted_top1'] for r in rr]))})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_max_abs':identity,'native_replay_max_abs':replay,'fit_eval_disjoint':not bool(set(train)&set(ix)),'fit_indices':train,'evaluation_indices':ix,'whole_model_forwards':actual,'expected_forwards':96*(1+2*len(MODES)),'prototype_fallback_positions':fallback,'finite':all(np.isfinite(r['native_KL']) and np.isfinite(r['effect_error']) for r in records),'test_evaluations':0}
 assert audits['hashes_unchanged'] and identity==0 and replay<1e-4 and audits['fit_eval_disjoint'] and actual==audits['expected_forwards'] and audits['finite'],audits
 torch.save({'fits':fits,'prototype_fits':protofits,'fit_indices':dict(fitids)},ART/f'{S}_fits.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'gate_log':gatelog,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(summary,audits,flush=True)
if __name__=='__main__':main()
