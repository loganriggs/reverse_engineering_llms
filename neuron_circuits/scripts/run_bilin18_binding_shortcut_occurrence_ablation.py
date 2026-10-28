"""H923 all-condition local value occurrence ablations on explicit QA."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model,final_logits,HEADS
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
S='bilin18_binding_shortcut_occurrence_ablation'
MODES=['identity','list_sender','list_recipient','event_sender','event_recipient','extra','all_names']
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'panel':ART/'bilin18_binding_repetition_role_screen_captures.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()}
 cfg={'experiment':'H923','hashes':hashes,'heads':HEADS,'modes':MODES,'design':'All256 H922 rows retained, no competence filtering. Single-reader ablations remove native persistent-value contribution at one of five name occurrences or all five. Readouts: event-second versus event-first preference, correct answer, individual logits/logprobs and native KL. No semantic binding claim from shortcut fidelity.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];assert len(ex)==256;model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 buckets=defaultdict(list)
 for i,e in enumerate(ex):buckets[len(e['ids'])].append(i)
 records=[];actual=0;identity=0.;replay=0.;deltaerr=0.
 for length,inds in sorted(buckets.items()):
  for st in range(0,len(inds),8):
   batch=inds[st:st+8];tokens=torch.tensor([ex[i]['ids'] for i in batch],device='cuda');native=final_logits(model,tokens).double().cpu();actual+=len(batch)
   for n,i in enumerate(batch):replay=max(replay,float((native[n]-data['native'][i]).abs().max()))
   for l,h in HEADS:
    for mode in MODES:
     dd=[]
     for i in batch:
      e=ex[i];roles=[] if mode=='identity' else (list(e['roles']) if mode=='all_names' else [mode]);A=data['routing'][i][l,h].double();V=data['bus'][i][:,h].double();coef=float(model.blocks[l].attn.lamb);delta=sum((-coef*A[e['roles'][r]]*V[e['roles'][r]] for r in roles),torch.zeros(128,dtype=torch.double));dd.append(delta)
      # Independent masked-value contraction audit.
      masked=V.clone()
      for r in roles:masked[e['roles'][r]]=0
      direct=coef*(A[:,None]*(masked-V)).sum(0);deltaerr=max(deltaerr,float((direct-delta).abs().max()))
     delta=torch.stack(dd).cuda()
     def edit(mod,args):
      x=args[0].clone();x[:,-1,h*128:(h+1)*128]+=delta.to(x);return (x,)
     handle=model.blocks[l].attn.c_proj.register_forward_pre_hook(edit)
     try:y=final_logits(model,tokens).double().cpu()
     finally:handle.remove()
     actual+=len(batch)
     if mode=='identity':identity=max(identity,float((native-y).abs().max()))
     for n,i in enumerate(batch):
      e=ex[i];second=1-e['sender'] if e['voice']=='sent_to' else e['sender'];ids=e['answer_ids'];v=ids[second];u=ids[1-second];good=ids[e['expected']];lp=native[n].log_softmax(-1);lq=y[n].log_softmax(-1);orderrole=mode
      if mode.startswith('event_'):orderrole='event_second' if e['roles'][mode]==max(e['roles']['event_sender'],e['roles']['event_recipient']) else 'event_first'
      if mode.startswith('list_'):orderrole='list_second' if e['roles'][mode]==max(e['roles']['list_sender'],e['roles']['list_recipient']) else 'list_first'
      records.append({k:e[k] for k in ['pair','split','voice','query','list_order','sender','extra','repeated_is_answer']}|{'index':i,'layer':l,'head':h,'mode':mode,'order_role':orderrole,'routing':float(sum(data['routing'][i][l,h][e['roles'][r]] for r in ([] if mode=='identity' else list(e['roles']) if mode=='all_names' else [mode]))),'native_second_margin':float(native[n,v]-native[n,u]),'second_margin_drop':float(native[n,v]-native[n,u]-y[n,v]+y[n,u]),'second_logit_change':float(y[n,v]-native[n,v]),'first_logit_change':float(y[n,u]-native[n,u]),'second_logprob_change':float(lq[v]-lp[v]),'first_logprob_change':float(lq[u]-lp[u]),'second_preferred':bool(y[n,v]>y[n,u]),'native_correct':int(native[n].argmax())==good,'edited_correct':int(y[n].argmax())==good,'KL':float((lp.exp()*(lp-lq)).sum())})
   print('DONE',length,st,actual,flush=True)
 groups=defaultdict(list)
 for r in records:groups[r['split'],r['layer'],r['head'],r['order_role']].append(r)
 summary=[]
 for key,rr in groups.items():summary.append(dict(zip(['split','layer','head','role'],key))|{'n':len(rr)}|{k:float(np.mean([r[k] for r in rr])) for k in ['routing','native_second_margin','second_margin_drop','second_logit_change','first_logit_change','second_logprob_change','first_logprob_change','second_preferred','native_correct','edited_correct','KL']})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_max_abs':identity,'native_replay_max_abs':replay,'masked_value_contraction_max_abs':deltaerr,'whole_model_forwards':actual,'expected_forwards':256*(1+3*len(MODES)),'finite':all(np.isfinite(r['KL']) and np.isfinite(r['second_margin_drop']) for r in records),'test_evaluations':0}
 assert audits['hashes_unchanged'] and identity==0 and replay<1e-4 and deltaerr<1e-10 and actual==audits['expected_forwards'] and audits['finite'],audits
 (ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
