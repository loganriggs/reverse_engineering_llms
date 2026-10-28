"""H820 frozen distinct-task competence, counterbalanced assignments/order/query."""
import json,re,time,math
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_broad_task_competence import continuation_id
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
S='bilin18_prefixed_ioi_capture'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 locked=ROOT/'configs/bilin18_prefixed_ioi_locked.json';names=json.loads(locked.read_text())['ioi_names']
 colors='red blue green black white gold pink brown gray orange yellow purple silver violet cyan tan'.split()
 places='Paris Rome London Berlin Tokyo Boston Denver Dallas Miami Austin Seattle Madrid Lisbon Dublin Moscow Vienna'.split()
 recipients='Jack Amy Ben Kate Sam Beth Joe Ann Luke Eve Mark Sue Nick May Dan Joy'.split()
 cfg={'experiment':'H820','design':'Three task families, prose and structured formats; sixteen items each, both value assignments, both fact orders, both queried entities. All contexts contain two distinct values. Exact answer label follows entity/value assignment. Single-token source values and answers, query-pair position/length alignment; complete eight-cell item gate.','tasks':['indirect_object'],'minimum_items':8,'minimum_candidate_accuracy':.75,'minimum_full_accuracy':.5,'minimum_both_queries_correct_rate':.5,'gate_scope':'Each task/format independently; require candidate>=.75 separately for query to first and second displayed fact. No correctness-based item removal.','frozen_mechanism':{'source_layer':2,'head':5,'source':'first displayed value','injection':'second displayed value','target_layer':6,'target_site':'query MLP','amplitudes':[0,.25,.5]},'future_contrast':'Compare query-to-first vs query-to-second at identical preceding facts; fixed first-value logprob and first-vs-second margin are primary response readouts, correctness secondary. Local source state is naturally query-invariant; do not duplicate it as an independent cross.','limitation':'Distinct from repeated identical-value copying, but still short two-fact retrieval; no general manifold inference.'}
 rows=[]
 for task in cfg['tasks']:
  for f in [0]:
   for i in range(32):
    ns=names[2*i:2*i+2]
    for order in [0,1]:
     vv=ns if order==0 else ns[::-1]
     for query in [0,1]:
      text=f'Yesterday, {vv[0]} and {vv[1]} went to the market. {vv[query]} gave a parcel to' if f==0 else f'{vv[0]} and {vv[1]} went to the store. {vv[query]} gave a book to'
      rows.append({'id':f'h820ioi-{f}-{i}-{order}-{query}','task':task,'format':f,'item':i,'assignment':0,'order':order,'query':query,'query_display_index':1-query,'text':text,'values':vv,'answer':vv[1-query]})
 cfg['design']='New fixed Yesterday prefix, market/parcel surface, thirty-two planned name pairs, both initial name orders and both giver choices. Correct recipient is other name. Complete four-cell item gate. Source values are initial two names; later giver mention is permitted.'
 cfg['future_contrast']='Giver-to-first versus giver-to-second; same preceding prefix, fixed first-name logprob and first-vs-second margin readout; candidate and full-vocabulary competence. No mechanism re-selection.'
 mp=ART/f'{S}_manifest.json';mp.write_text(json.dumps(rows,indent=2)+'\n');paths={'locked':locked,'manifest':mp,'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg['hashes']=hashes;(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);ex={};excluded=[]
 for r in rows:
  e=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True);pos=[]
  for v in r['values']:
   spans=list(re.finditer(r'\b'+re.escape(v)+r'\b',r['text']))
   if len(spans)<1:break
   a,b=spans[0].span();ix=[k for k,(x,y) in enumerate(e['offset_mapping']) if x<b and y>a]
   if len(ix)!=1:break
   pos.append(ix[0])
  ci=[continuation_id(tok,r['text'],' '+v) for v in r['values']]
  if len(pos)!=2 or None in ci or len(set(ci))<2:excluded.append({'id':r['id'],'reason':'source/answer single-token validity'});continue
  ex[r['id']]={**r,'ids':e['input_ids'],'positions':pos,'candidate_ids':ci}
 groups=defaultdict(list)
 for e in ex.values():groups[e['task'],e['format'],e['item']].append(e)
 valid=[]
 for (task,f,i),ee in groups.items():
  ok=len(ee)==4
  for a in [0]:
   for o in [0,1]:
    pp=[e for e in ee if e['assignment']==a and e['order']==o];ok=ok and len(pp)==2 and len({(len(e['ids']),tuple(e['positions'])) for e in pp})==1
  if not ok:excluded.append({'task':task,'format':f,'item':i,'reason':'incomplete four cells or paired-query alignment'});continue
  valid.extend(ee)
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 buckets=defaultdict(list)
 for e in valid:buckets[len(e['ids'])].append(e)
 records=[];captures={};attn=model.blocks[2].attn;orig=attn.squared_attention;saved={};batch=[]
 def capture(q,k,v,q2,k2):
  saved['v']=torch.stack([v[i,e['positions'][0],5] for i,e in enumerate(batch)]).cpu();return orig(q,k,v,q2,k2)
 attn.squared_attention=capture
 try:
  for ee in buckets.values():
   for lo in range(0,len(ee),12):
    batch=ee[lo:lo+12];out,_=run(model,torch.tensor([e['ids'] for e in batch],device='cuda'))
    for n,e in enumerate(batch):
     p=out[n].cpu();t=e['query_display_index'];records.append({k:e[k] for k in ['id','task','format','item','assignment','order','query','query_display_index']}|{'candidate_correct':int(p[e['candidate_ids']].argmax())==t,'full_correct':int(p.argmax())==e['candidate_ids'][t],'margin_first_minus_second':float(p[e['candidate_ids'][0]]-p[e['candidate_ids'][1]])});captures[e['id']]={'value':saved['v'][n],'logits':p,'example':e}
 finally:attn.squared_attention=orig
 summary=[]
 for task in cfg['tasks']:
  for f in [0]:
   rr=[r for r in records if r['task']==task and r['format']==f];items=len({r['item'] for r in rr});mean=lambda x:sum(x)/len(x) if x else 0;acc=mean([r['candidate_correct'] for r in rr]);full=mean([r['full_correct'] for r in rr]);byq=[mean([r['candidate_correct'] for r in rr if r['query_display_index']==q]) for q in [0,1]];pairs=defaultdict(list)
   for r in rr:pairs[r['item'],r['assignment'],r['order']].append(r['candidate_correct'])
   paired=mean([all(v) for v in pairs.values()]);summary.append({'task':task,'format':f,'items':items,'candidate_accuracy':acc,'full_accuracy':full,'candidate_by_display_query':byq,'both_queries_correct_rate':paired,'eligible':items>=8 and min(byq)>=.75 and full>=.5 and paired>=.5})
 idx={(r['task'],r['format'],r['item'],r['assignment'],r['order'],r['query']):r['id'] for r in records};err=max(float((captures[r['id']]['value']-captures[idx[r['task'],r['format'],r['item'],r['assignment'],r['order'],1-r['query']]]['value']).abs().max()) for r in records)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'paired_query_source_value_max_abs':err,'finite':all(math.isfinite(r['margin_first_minus_second']) for r in records),'whole_model_forwards':len(records),'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and err==0
 torch.save(captures,ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'excluded':excluded,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits)
 for r in summary:print(r)
if __name__=='__main__':main()
