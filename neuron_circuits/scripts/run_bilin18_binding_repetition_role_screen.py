"""H922 fixed explicit-fact screen: extra repetition crossed with queried role."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model,final_logits,HEADS,TOKENIZER,continuation_id,PAIRS
from transformers import AutoTokenizer
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
S='bilin18_binding_repetition_role_screen'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','prior':ART/'bilin18_binding_routing_rule_transfer.json'};hashes={k:sha(p) for k,p in paths.items()};pairs=[p['sources'] for p in PAIRS if p['domain']=='binding']
 cfg={'experiment':'H922','hashes':hashes,'pairs':pairs,'heads':HEADS,'design':'Explicit fact QA: listorder, sender identity, extra-mentioned identity, active/passive voice, sender/recipient query independently crossed. Every name occurs in list and event; extra mention gives one name a third occurrence. Same final Answer: suffix. No prompt adaptation or correctness filtering. First4 discovery,last4 transfer names, familiar names. Native screen only; earlier three-occurrence rule not directly applied to this five-occurrence format.','gate':{'min_pairs':4,'min_both_sender_endpoints_binary':.75,'min_top1':.5}};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);ex=[]
 for pair,names in enumerate(pairs):
  tids=[tok.encode(' '+n,add_special_tokens=False) for n in names];assert all(len(t)==1 for t in tids)
  for order in [0,1]:
   a,b=names if order==0 else names[::-1]
   for sender in [0,1]:
    for extra in [0,1]:
     for voice in ['active','passive']:
      event=f'{names[sender]} sent a letter to {names[1-sender]}.' if voice=='active' else f'{names[1-sender]} received a letter from {names[sender]}.'
      # received-from is a lexical argument reversal, not grammatical passive; label preserved below as reverse.
      for query in ['sender','recipient']:
       question='Who sent the letter?' if query=='sender' else 'Who received the letter?';text=f'Yesterday, {a} and {b} met. {event} {names[extra]} was smiling.\nQuestion: {question}\nAnswer:';ids=tok.encode(text,add_special_tokens=False);ans=[continuation_id(tok,text,' '+n) for n in names];assert all(v is not None for v in ans)
       pos=[[p for p,t in enumerate(ids) if t==ti[0]] for ti in tids];assert len(pos[extra])==3 and len(pos[1-extra])==2
       roles={'list_sender':pos[sender][0],'list_recipient':pos[1-sender][0],'event_sender':pos[sender][1],'event_recipient':pos[1-sender][1],'extra':pos[extra][2]};expected=sender if query=='sender' else 1-sender
       ex.append(dict(pair=pair,split='discovery' if pair<4 else 'transfer',list_order=order,sender=sender,extra=extra,voice='sent_to' if voice=='active' else 'received_from',query=query,repeated_is_answer=extra==expected,text=text,ids=ids,answer_ids=ans,expected=expected,roles=roles))
 assert len(ex)==256
 cfg['design']=cfg['design'].replace('active/passive voice','sent-to/received-from argument order');(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');(ART/f'{S}_manifest.json').write_text(json.dumps({'examples':ex},indent=2)+'\n')
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 stash={};originals={};handles=[]
 for l,h in HEADS:
  fn=model.blocks[l].attn.squared_attention;originals[l]=fn
  def capture(q,k,v,q2,k2,l=l,h=h,fn=fn):
   stash[l,h]=((torch.einsum('bhd,bthd->bht',q[:,-1],k)/128)*(torch.einsum('bhd,bthd->bht',q2[:,-1],k2)/128))[:,h].detach().cpu();return fn(q,k,v,q2,k2)
  model.blocks[l].attn.squared_attention=capture
 def buscap(mod,args,out):stash['bus']=out[1].detach().cpu()
 handles.append(model.blocks[0].register_forward_hook(buscap));buckets=defaultdict(list)
 for i,e in enumerate(ex):buckets[len(e['ids'])].append(i)
 native={};bus={};routing={};records=[];actual=0
 try:
  for length,inds in sorted(buckets.items()):
   for st in range(0,len(inds),8):
    batch=inds[st:st+8];y=final_logits(model,torch.tensor([ex[i]['ids'] for i in batch],device='cuda')).double().cpu();actual+=len(batch)
    for n,i in enumerate(batch):
     e=ex[i];native[i]=y[n];bus[i]=stash['bus'][n].clone();routing[i]={head:stash[head][n].clone() for head in HEADS};good=e['answer_ids'][e['expected']];bad=e['answer_ids'][1-e['expected']]
     records.append({k:e[k] for k in ['pair','split','list_order','sender','extra','voice','query','repeated_is_answer']}|{'index':i,'margin':float(y[n,good]-y[n,bad]),'top1':int(y[n].argmax())==good,'top_token':tok.decode([int(y[n].argmax())])})
   print('DONE',length,actual,flush=True)
 finally:
  for h in handles:h.remove()
  for l,fn in originals.items():model.blocks[l].attn.squared_attention=fn
 groups=defaultdict(list)
 for r in records:groups[r['split'],r['voice'],r['query'],r['repeated_is_answer']].append(r)
 cells=[]
 for key,rr in groups.items():
  by=defaultdict(list)
  for r in rr:by[r['pair'],r['list_order']].append(r)
  assert all(len(v)==2 for v in by.values());binary=float(np.mean([all(r['margin']>0 for r in v) for v in by.values()]));top=float(np.mean([r['top1'] for r in rr]));cells.append(dict(zip(['split','voice','query','repeated_is_answer'],key))|{'n':len(rr),'both_binary':binary,'top1':top,'passes':binary>=.75 and top>=.5})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'whole_model_forwards':actual,'expected_forwards':256,'test_evaluations':0,'finite':all(np.isfinite(r['margin']) for r in records),'all_name_occurrence_counts_verified':True}
 assert audits['hashes_unchanged'] and actual==256 and audits['finite'],audits
 torch.save({'examples':ex,'native':native,'bus':bus,'routing':routing},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'competence':cells,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(cells,audits,flush=True)
if __name__=='__main__':main()
