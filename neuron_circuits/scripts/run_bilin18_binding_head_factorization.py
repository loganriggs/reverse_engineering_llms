"""H925 all-head factorization at exploratory localized layers14/16."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
S='bilin18_binding_head_factorization';LAYERS=[14,16];MODES=['routing','mixed_values','bus','current_values','full']
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'panel':ART/'bilin18_binding_repetition_role_screen_captures.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H925','hashes':hashes,'layers':LAYERS,'heads':list(range(9)),'modes':MODES,'design':'All128 transfer rows, both matched contrasts, no competence filtering. Layers selected exploratorily by H924, allheads retained. Single-head final-token output edits: routing-only,mixedvalues-only,persistentbus-only,currentvalues-only,full donor. Exact local cross-term audited; no claim downstream effects add.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];ix=[i for i,e in enumerate(ex) if e['split']=='transfer'];keys=['pair','list_order','sender','extra','voice','query'];lookup={tuple(e[k] for k in keys):i for i,e in enumerate(ex)};donors={}
 for i in ix:
  for c in ['event_swap','same_fact_reword']:
   e=dict(ex[i])
   if c=='event_swap':e['sender']=1-e['sender']
   else:e['voice']='sent_to' if e['voice']=='received_from' else 'received_from'
   donors[i,c]=lookup[tuple(e[k] for k in keys)];assert donors[i,c] in ix
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 stash={};cache={};originals={};handles=[]
 for l in LAYERS:
  fn=model.blocks[l].attn.squared_attention;originals[l]=fn
  def capture(q,k,v,q2,k2,l=l,fn=fn):
   stash[l,'A']=((torch.einsum('bhd,bthd->bht',q[:,-1],k)/128)*(torch.einsum('bhd,bthd->bht',q2[:,-1],k2)/128)).detach().cpu();stash[l,'V']=v.detach().cpu();return fn(q,k,v,q2,k2)
  model.blocks[l].attn.squared_attention=capture
  def pre(mod,args,l=l):stash[l,'H']=args[0][:,-1].detach().cpu().reshape(-1,9,128).clone()
  handles.append(model.blocks[l].attn.c_proj.register_forward_pre_hook(pre))
 native={};actual=0;replay=0.;averr=0.;identity=0.;algerr=0.;records=[];geometry=[]
 try:
  for st in range(0,len(ix),16):
   batch=ix[st:st+16];y=final_logits(model,torch.tensor([ex[i]['ids'] for i in batch],device='cuda')).double().cpu();actual+=len(batch)
   for n,i in enumerate(batch):
    native[i]=y[n];replay=max(replay,float((y[n]-data['native'][i]).abs().max()))
    for key,v in stash.items():cache[i,*key]=v[n].double().clone()
  for l,fn in originals.items():model.blocks[l].attn.squared_attention=fn
  for h in handles:h.remove()
  handles=[]
  deltas={}
  for i in ix:
   for c in ['event_swap','same_fact_reword']:
    j=donors[i,c]
    for l in LAYERS:
     for h in range(9):
      A=cache[i,l,'A'][h];B=cache[j,l,'A'][h];V=cache[i,l,'V'][:,h];W=cache[j,l,'V'][:,h];base=(A[:,None]*V).sum(0);full=(B[:,None]*W).sum(0)-base;route=((B-A)[:,None]*V).sum(0);value=(A[:,None]*(W-V)).sum(0);db=float(model.blocks[l].attn.lamb)*(data['bus'][j][:,h]-data['bus'][i][:,h]).double();bus=(A[:,None]*db).sum(0);current=(A[:,None]*(W-V-db)).sum(0);cross=((B-A)[:,None]*(W-V)).sum(0)
      algerr=max(algerr,float((full-route-value-cross).abs().max()),float((value-bus-current).abs().max()));averr=max(averr,float((base-cache[i,l,'H'][h]).norm()/cache[i,l,'H'][h].norm().clamp_min(1e-12)))
      for mode,v in [('routing',route),('mixed_values',value),('bus',bus),('current_values',current),('full',cache[j,l,'H'][h]-cache[i,l,'H'][h])]:deltas[i,c,l,h,mode]=v
      geometry.append(dict(index=i,contrast=c,layer=l,head=h,cross_norm=float(cross.norm()),full_norm=float(full.norm())))
  for st in range(0,len(ix),16):
   batch=ix[st:st+16];tokens=torch.tensor([ex[i]['ids'] for i in batch],device='cuda')
   for l in LAYERS:
    for h in range(9):
     jobs=[(c,m) for c in ['event_swap','same_fact_reword'] for m in MODES]+([('self','identity')] if st==0 else [])
     for c,mode in jobs:
      delta=torch.zeros(len(batch),128,device='cuda') if mode=='identity' else torch.stack([deltas[i,c,l,h,mode] for i in batch]).cuda()
      def edit(mod,args):
       x=args[0].clone();x[:,-1,h*128:(h+1)*128]+=delta.to(x);return (x,)
      handle=model.blocks[l].attn.c_proj.register_forward_pre_hook(edit)
      try:y=final_logits(model,tokens).double().cpu()
      finally:handle.remove()
      actual+=len(batch)
      if mode=='identity':identity=max(identity,max(float((y[n]-native[i]).abs().max()) for n,i in enumerate(batch)));continue
      for n,i in enumerate(batch):
       e=ex[i];second=1-e['sender'] if e['voice']=='sent_to' else e['sender'];v=e['answer_ids'][second];u=e['answer_ids'][1-second];lp=native[i].log_softmax(-1);lq=y[n].log_softmax(-1)
       records.append({k:e[k] for k in ['pair','voice','query','sender','extra','list_order']}|{'index':i,'contrast':c,'layer':l,'head':h,'mode':mode,'margin_drop':float(native[i][v]-native[i][u]-y[n,v]+y[n,u]),'KL':float((lp.exp()*(lp-lq)).sum()),'second_preferred':bool(y[n,v]>y[n,u])})
   print('DONE',st,actual,flush=True)
 finally:
  for handle in handles:handle.remove()
  for l,fn in originals.items():model.blocks[l].attn.squared_attention=fn
 groups=defaultdict(list)
 for r in records:groups[r['contrast'],r['layer'],r['head'],r['mode']].append(r)
 summary=[]
 for key,rr in groups.items():summary.append(dict(zip(['contrast','layer','head','mode'],key))|{'mean_drop':float(np.mean([r['margin_drop'] for r in rr])),'RMS':float(np.mean([r['margin_drop']**2 for r in rr])**.5),'KL':float(np.mean([r['KL'] for r in rr])),'second_preferred':float(np.mean([r['second_preferred'] for r in rr]))})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_max_abs':identity,'native_replay_max_abs':replay,'AV_relative_error':averr,'factorization_max_abs':algerr,'whole_model_forwards':actual,'expected_forwards':128+128*2*9*2*len(MODES)+16*18,'test_evaluations':0,'finite':all(np.isfinite(r['KL']) and np.isfinite(r['margin_drop']) for r in records)}
 assert audits['hashes_unchanged'] and identity==0 and replay<1e-4 and averr<1e-4 and algerr<1e-10 and actual==audits['expected_forwards'] and audits['finite'],audits
 torch.save(cache,ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'geometry':geometry,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
