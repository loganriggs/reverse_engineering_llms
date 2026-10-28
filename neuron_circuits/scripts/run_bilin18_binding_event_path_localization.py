"""H924 matched contrast localization across all attention/MLP final-token writes."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
S='bilin18_binding_event_path_localization'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'panel':ART/'bilin18_binding_repetition_role_screen_captures.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()}
 cfg={'experiment':'H924','hashes':hashes,'design':'Capture native final-token attention/MLP output at all18layers for256 H922 rows. Evaluate all128 transfer rows; no competence filter. Donor1 changes sender only (event name swap); donor2 changes sent-to/received-from wording only (same fact, reversed event arguments). Patch one attention or MLP output at finaltoken; rest live. Native/self and token alignment audits. Existing names, not pristine test.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];keys=['pair','list_order','sender','extra','voice','query'];lookup={tuple(e[k] for k in keys):i for i,e in enumerate(ex)};donors={};changed={};ix=[i for i,e in enumerate(ex) if e['split']=='transfer'];assert len(ix)==128
 for i in ix:
  e=ex[i]
  for kind in ['event_swap','same_fact_reword']:
   d=dict(e)
   if kind=='event_swap':d['sender']=1-e['sender']
   else:d['voice']='received_from' if e['voice']=='sent_to' else 'sent_to'
   j=lookup[tuple(d[k] for k in keys)];donors[i,kind]=j;other=ex[j];assert len(e['ids'])==len(other['ids']) and e['ids'][-1]==other['ids'][-1];diff=[p for p,(a,b) in enumerate(zip(e['ids'],other['ids'])) if a!=b];changed[i,kind]=diff
   if kind=='event_swap':assert set(diff)=={e['roles']['event_sender'],e['roles']['event_recipient']}
   assert e['ids'][:min(e['roles']['event_sender'],e['roles']['event_recipient'])]==other['ids'][:min(other['roles']['event_sender'],other['roles']['event_recipient'])]
   end=max(e['roles']['event_sender'],e['roles']['event_recipient']);assert e['ids'][end+1:]==other['ids'][end+1:]
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 cache={};stash={};state={'capture':True,'target':None,'donor':None};handles=[]
 def hook(k,part):
  def fn(mod,args,out):
   x=out[0] if part=='attention' else out
   if state['capture']:stash[k,part]=x[:,-1].detach().cpu().clone();return
   if state['target']!=(k,part):return
   y=x.clone();y[:,-1]=state['donor'].to(y)
   return (y,out[1]) if part=='attention' else y
  return fn
 for k,b in enumerate(model.blocks):handles.append(b.attn.register_forward_hook(hook(k,'attention')));handles.append(b.mlp.register_forward_hook(hook(k,'mlp')))
 buckets=defaultdict(list)
 for i,e in enumerate(ex):buckets[len(e['ids'])].append(i)
 native={};records=[];actual=0;replay=0.;identity=0.;selfcount=0
 try:
  for length,inds in sorted(buckets.items()):
   for st in range(0,len(inds),8):
    batch=inds[st:st+8];y=final_logits(model,torch.tensor([ex[i]['ids'] for i in batch],device='cuda')).double().cpu();actual+=len(batch)
    for n,i in enumerate(batch):
     native[i]=y[n];replay=max(replay,float((y[n]-data['native'][i]).abs().max()))
     for key,v in stash.items():cache[i,*key]=v[n].clone()
  state['capture']=False
  for st in range(0,len(ix),8):
   batch=ix[st:st+8];tokens=torch.tensor([ex[i]['ids'] for i in batch],device='cuda')
   for layer in range(18):
    for part in ['attention','mlp']:
     state['target']=layer,part
     for kind in (['self','event_swap','same_fact_reword'] if st==0 else ['event_swap','same_fact_reword']):
      js=batch if kind=='self' else [donors[i,kind] for i in batch];state['donor']=torch.stack([cache[j,layer,part] for j in js]);y=final_logits(model,tokens).double().cpu();actual+=len(batch)
      if kind=='self':
       selfcount+=len(batch);identity=max(identity,max(float((y[n]-native[i]).abs().max()) for n,i in enumerate(batch)));continue
      for n,(i,j) in enumerate(zip(batch,js)):
       e=ex[i];second=1-e['sender'] if e['voice']=='sent_to' else e['sender'];ids=e['answer_ids'];v=ids[second];u=ids[1-second];margin=lambda z:float(z[v]-z[u]);lp=native[i].log_softmax(-1);lq=y[n].log_softmax(-1)
       records.append({k:e[k] for k in ['pair','voice','query','list_order','sender','extra']}|{'index':i,'donor':j,'contrast':kind,'layer':layer,'part':part,'native_margin':margin(native[i]),'full_donor_drop':margin(native[i])-margin(native[j]),'patch_drop':margin(native[i])-margin(y[n]),'KL':float((lp.exp()*(lp-lq)).sum()),'second_preferred':bool(y[n,v]>y[n,u]),'task_top1':int(y[n].argmax())==ids[e['expected']]})
   print('DONE',st,actual,flush=True)
 finally:
  for handle in handles:handle.remove()
 groups=defaultdict(list)
 for r in records:groups[r['contrast'],r['layer'],r['part']].append(r)
 summary=[]
 for key,rr in groups.items():
  d=np.array([r['full_donor_drop'] for r in rr]);p=np.array([r['patch_drop'] for r in rr]);summary.append(dict(zip(['contrast','layer','part'],key))|{'mean_full_donor_drop':float(d.mean()),'mean_patch_drop':float(p.mean()),'patch_RMS':float(np.mean(p*p)**.5),'error_to_full_ratio':float(np.mean((p-d)**2)**.5/np.mean(d*d)**.5),'KL':float(np.mean([r['KL'] for r in rr])),'second_preferred':float(np.mean([r['second_preferred'] for r in rr]))})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_max_abs':identity,'native_replay_max_abs':replay,'self_control_forwards':selfcount,'whole_model_forwards':actual,'expected_forwards':256+128*36*2+8*36,'event_swap_only_event_names':True,'same_length_finaltoken_prefix_suffix_verified':True,'finite':all(np.isfinite(r['KL']) and np.isfinite(r['patch_drop']) for r in records),'test_evaluations':0}
 assert audits['hashes_unchanged'] and identity==0 and replay<1e-4 and actual==audits['expected_forwards'] and audits['finite'],audits
 torch.save(cache,ART/f'{S}_native_writes.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
