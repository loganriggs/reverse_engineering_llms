"""H920 role/order counterbalance and local persistent-value occurrence ablations."""
from run_bilin18_multidomain_matched_competence import ROOT,ART,sha,SNAPSHOT,load_full_model,final_logits,TOKENIZER,continuation_id,PAIRS
from run_bilin18_binding_route_value_factorial import HEADS
from transformers import AutoTokenizer
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
S='bilin18_binding_role_order_panel'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};pairs=[p['sources'] for p in PAIRS if p['domain']=='binding']
 cfg={'experiment':'H920','hashes':hashes,'pairs':pairs,'heads':HEADS,'design':'Existing8 name pairs. Task crosses listorder0/1 and sender0/1 in two frozen templates. Copy controls have no listorder and are counted once. All eligible outcomes retained, no competence selection. Local occurrence ablations remove only lamb*A[p]*bus[p,h] at one reader; no global value-stream edit. First4 discovery,last4 transfer, familiar names.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);ex=[]
 for pair,names in enumerate(pairs):
  tids=[tok.encode(' '+n,add_special_tokens=False) for n in names];assert all(len(t)==1 for t in tids)
  for fmt in ['A','B']:
   for sender in [0,1]:
    for family,orders in [('task',[0,1]),('copy',[0])]:
     for order in orders:
      a,b=names if order==0 else names[::-1];w=names[sender]
      text=(f'Yesterday, {a} and {b} met. {w} sent a letter to' if fmt=='A' else f'Earlier, {a} and {b} met. {w} gave a book to') if family=='task' else (f'word {w}; word {w}; word' if fmt=='A' else f'item {w}, item {w}, item')
      ids=tok.encode(text,add_special_tokens=False);answers=[continuation_id(tok,text,' '+n) for n in names];assert all(v is not None for v in answers)
      positions=[[p for p,t in enumerate(ids) if t==ti[0]] for ti in tids]
      if family=='task':assert len(positions[sender])==2 and len(positions[1-sender])==1;roles={'list_sender':positions[sender][0],'list_other':positions[1-sender][0],'sender_repeat':positions[sender][1]}
      else:assert len(positions[sender])==2;roles={'copy_first':positions[sender][0],'copy_second':positions[sender][1]}
      ex.append(dict(pair=pair,split='discovery' if pair<4 else 'transfer',format=fmt,sender=sender,list_order=order,family=family,text=text,ids=ids,answer_ids=answers,expected=1-sender if family=='task' else sender,roles=roles))
 assert len(ex)==96
 (ART/f'{S}_manifest.json').write_text(json.dumps({'examples':ex},indent=2)+'\n')
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 stash={};originals={};handles=[]
 for l,h in HEADS:
  att=model.blocks[l].attn;fn=att.squared_attention;originals[l]=fn
  def capture(q,k,v,q2,k2,l=l,h=h,fn=fn):
   A=(torch.einsum('bhd,bthd->bht',q[:,-1],k)/128)*(torch.einsum('bhd,bthd->bht',q2[:,-1],k2)/128);stash[l,h]=A[:,h].detach().cpu();return fn(q,k,v,q2,k2)
  att.squared_attention=capture
 def bushook(mod,args,out):stash['bus']=out[1].detach().cpu()
 handles.append(model.blocks[0].register_forward_hook(bushook));buckets=defaultdict(list)
 for i,e in enumerate(ex):buckets[len(e['ids']),e['family']].append(i)
 native={};bus={};routing={};records=[];actual=0;identity=0.
 try:
  for (length,family),inds in sorted(buckets.items()):
   for st in range(0,len(inds),8):
    batch=inds[st:st+8];tokens=torch.tensor([ex[i]['ids'] for i in batch],device='cuda');y=final_logits(model,tokens).double().cpu();actual+=len(batch)
    for n,i in enumerate(batch):
     native[i]=y[n];bus[i]=stash['bus'][n].clone();routing[i]={head:stash[head][n].clone() for head in HEADS}
    for l,h in HEADS:
     for role in ['identity']+list(ex[batch[0]]['roles']):
      dd=[]
      for i in batch:
       if role=='identity':dd.append(torch.zeros(128))
       else:p=ex[i]['roles'][role];dd.append(-float(model.blocks[l].attn.lamb)*routing[i][l,h][p].double()*bus[i][p,h].double())
      delta=torch.stack(dd).cuda()
      def edit(mod,args):
       x=args[0].clone();x[:,-1,h*128:(h+1)*128]+=delta.to(x);return (x,)
      handle=model.blocks[l].attn.c_proj.register_forward_pre_hook(edit)
      try:z=final_logits(model,tokens).double().cpu()
      finally:handle.remove()
      actual+=len(batch)
      if role=='identity':identity=max(identity,float((z-y).abs().max()))
      for n,i in enumerate(batch):
       e=ex[i];good=e['answer_ids'][e['expected']];bad=e['answer_ids'][1-e['expected']];ln=y[n].log_softmax(-1);le=z[n].log_softmax(-1)
       records.append({k:e[k] for k in ['pair','split','format','sender','list_order','family']}|{'index':i,'layer':l,'head':h,'role':role,'routing':0. if role=='identity' else float(routing[i][l,h][e['roles'][role]]),'native_margin':float(y[n,good]-y[n,bad]),'margin_drop':float(y[n,good]-y[n,bad]-z[n,good]+z[n,bad]),'correct_logit_change':float(z[n,good]-y[n,good]),'other_logit_change':float(z[n,bad]-y[n,bad]),'correct_logprob_change':float(le[good]-ln[good]),'other_logprob_change':float(le[bad]-ln[bad]),'KL':float((ln.exp()*(ln-le)).sum()),'native_top1':int(y[n].argmax())==good,'edited_top1':int(z[n].argmax())==good})
    print('DONE',family,length,st,actual,flush=True)
 finally:
  for handle in handles:handle.remove()
  for l,fn in originals.items():model.blocks[l].attn.squared_attention=fn
 cells=[]
 for split in ['discovery','transfer']:
  for family in ['task','copy']:
   for fmt in ['A','B']:
    for order in ([0,1] if family=='task' else [0]):
     ii=[i for i,e in enumerate(ex) if (e['split'],e['family'],e['format'],e['list_order'])==(split,family,fmt,order)];scores={i:float(native[i][ex[i]['answer_ids'][ex[i]['expected']]]-native[i][ex[i]['answer_ids'][1-ex[i]['expected']]]) for i in ii};by=defaultdict(list)
     for i in ii:by[ex[i]['pair']].append(i)
     both=float(np.mean([all(scores[i]>0 for i in js) for js in by.values()]));top=float(np.mean([int(native[i].argmax())==ex[i]['answer_ids'][ex[i]['expected']] for i in ii]));cells.append(dict(split=split,family=family,format=fmt,list_order=order,both_binary=both,top1=top,passes=len(by)>=4 and both>=.75 and top>=.5))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_max_abs':identity,'whole_model_forwards':actual,'expected_forwards':96+64*12+32*9,'test_evaluations':0,'finite':all(np.isfinite(r['margin_drop']) and np.isfinite(r['KL']) for r in records)}
 assert audits['hashes_unchanged'] and identity==0 and actual==audits['expected_forwards'] and audits['finite'],audits
 torch.save({'examples':ex,'native':native,'bus':bus,'routing':routing},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'competence':cells,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(cells,audits,flush=True)
if __name__=='__main__':main()
