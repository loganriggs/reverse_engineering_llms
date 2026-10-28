"""H921: discovery-fitted persistent-value routing coefficients."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model,final_logits,HEADS
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
S='bilin18_binding_routing_rule_transfer'
MODES=['identity','zero','own_mean','other_format','pooled','occurrence','wrong_occurrence','sign_flip']
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'panel':ART/'bilin18_binding_role_order_panel_captures.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()}
 cfg={'experiment':'H921','hashes':hashes,'modes':MODES,'heads':HEADS,'design':'Fit role coefficient means on first4 discovery name pairs only, balanced sender/list orders. Transfer other4 names. Native recipient bus values and current-value contributions retained. Single readers and joint3. own_mean template-specific; other_format uses other template; pooled averages templates; occurrence additionally ties both initial-list roles; wrong_occurrence swaps list_other/sender_repeat (copy swaps occurrences); sign_flip negates pooled. zero removes all selected name bus contributions. No claim of whole-head replacement.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];fits={};fitmeta=[]
 for family in ['task','copy']:
  for fmt in ['A','B']:
   jj=[i for i,e in enumerate(ex) if e['split']=='discovery' and e['family']==family and e['format']==fmt]
   for head in HEADS:
    for role in ex[jj[0]]['roles']:
     fits[family,fmt,*head,role]=float(np.mean([float(data['routing'][i][head][ex[i]['roles'][role]]) for i in jj]));fitmeta.append({'family':family,'format':fmt,'head':head,'role':role,'mean':fits[family,fmt,*head,role],'fit_indices':jj})
 ix=[i for i,e in enumerate(ex) if e['split']=='transfer'];assert len(ix)==48
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 buckets=defaultdict(list)
 for i in ix:buckets[len(ex[i]['ids']),ex[i]['family']].append(i)
 originals={};hooks=[];live={};state={'mode':'identity','targets':[],'batch':[]};identity=0.;replay=0.;actual=0;records=[];coefferrors=[]
 def coeff(e,head,role,mode):
  family=e['family'];fmt=e['format'];own=lambda f,r:fits[family,f,*head,r];pool=lambda r:(own('A',r)+own('B',r))/2
  if mode=='zero':return 0.
  if mode=='own_mean':return own(fmt,role)
  if mode=='other_format':return own('B' if fmt=='A' else 'A',role)
  if mode=='pooled':return pool(role)
  if mode=='occurrence':return (pool('list_sender')+pool('list_other'))/2 if family=='task' and role.startswith('list_') else pool(role)
  if mode=='wrong_occurrence':
   swap={'list_other':'sender_repeat','sender_repeat':'list_other','list_sender':'list_sender','copy_first':'copy_second','copy_second':'copy_first'};return pool(swap[role])
  if mode=='sign_flip':return -pool(role)
  raise ValueError(mode)
 for l,h in HEADS:
  att=model.blocks[l].attn;fn=att.squared_attention;originals[l]=fn
  def capture(q,k,v,q2,k2,l=l,h=h,fn=fn):
   live[l,h]=((torch.einsum('bhd,bthd->bht',q[:,-1],k)/128)*(torch.einsum('bhd,bthd->bht',q2[:,-1],k2)/128))[:,h];return fn(q,k,v,q2,k2)
  att.squared_attention=capture
  def edit(mod,args,l=l,h=h):
   if state['mode']=='identity' or (l,h) not in state['targets']:return
   changes=[]
   for n,i in enumerate(state['batch']):
    e=ex[i];delta=torch.zeros(128,device='cuda',dtype=torch.double)
    for role,p in e['roles'].items():delta+=(coeff(e,(l,h),role,state['mode'])-live[l,h][n,p].double())*data['bus'][i][p,h].double().cuda()
    changes.append(delta*float(model.blocks[l].attn.lamb))
   x=args[0].clone();x[:,-1,h*128:(h+1)*128]+=torch.stack(changes).to(x);return (x,)
  hooks.append(att.c_proj.register_forward_pre_hook(edit))
 try:
  for (length,family),inds in sorted(buckets.items()):
   for st in range(0,len(inds),8):
    batch=inds[st:st+8];state.update(mode='identity',batch=batch);tokens=torch.tensor([ex[i]['ids'] for i in batch],device='cuda');native=final_logits(model,tokens).double().cpu();actual+=len(batch)
    for n,i in enumerate(batch):replay=max(replay,float((native[n]-data['native'][i]).abs().max()))
    for targets in [[h] for h in HEADS]+[HEADS]:
     target='joint' if len(targets)>1 else f'L{targets[0][0]}H{targets[0][1]}'
     for mode in MODES:
      state.update(mode=mode,targets=targets);y=final_logits(model,tokens).double().cpu();actual+=len(batch)
      if mode=='identity':identity=max(identity,float((native-y).abs().max()))
      for n,i in enumerate(batch):
       e=ex[i];good=e['answer_ids'][e['expected']];bad=e['answer_ids'][1-e['expected']];lp=native[n].log_softmax(-1);lq=y[n].log_softmax(-1)
       records.append({k:e[k] for k in ['pair','family','format','sender','list_order']}|{'index':i,'target':target,'mode':mode,'native_margin':float(native[n,good]-native[n,bad]),'margin_change':float(y[n,good]-y[n,bad]-native[n,good]+native[n,bad]),'KL':float((lp.exp()*(lp-lq)).sum()),'native_top1':int(native[n].argmax())==good,'top1':int(y[n].argmax())==good})
    print('DONE',family,length,st,actual,flush=True)
 finally:
  for hook in hooks:hook.remove()
  for l,fn in originals.items():model.blocks[l].attn.squared_attention=fn
 groups=defaultdict(list)
 for r in records:groups[r['family'],r['target'],r['mode']].append(r)
 summary=[]
 for (family,target,mode),rr in groups.items():
  zero=groups[family,target,'zero'];rms=lambda rows:float(np.mean([r['margin_change']**2 for r in rows])**.5);kl=float(np.mean([r['KL'] for r in rr]));base=float(np.mean([r['KL'] for r in zero]));summary.append(dict(family=family,target=target,mode=mode,margin_RMS=rms(rr),error_ratio=rms(rr)/rms(zero),KL=kl,KL_ratio=kl/base,top1=float(np.mean([r['top1'] for r in rr]))))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_max_abs':identity,'native_replay_max_abs':replay,'whole_model_forwards':actual,'expected_forwards':48*(1+4*len(MODES)),'finite':all(np.isfinite(r['KL_ratio']) for r in summary),'test_evaluations':0}
 assert audits['hashes_unchanged'] and identity==0 and replay<1e-4 and actual==audits['expected_forwards'] and audits['finite'],audits
 (ART/f'{S}.json').write_text(json.dumps({'config':cfg,'fits':fitmeta,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
