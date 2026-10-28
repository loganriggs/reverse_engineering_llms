"""H807 native key-identity by value-identity factorial for frozen L2H5 edge."""
import json,time,re,math
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
S='bilin18_fresh_bird_capture'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');keys='red blue green black white gold pink brown gray orange yellow purple silver violet cyan tan'.split();lock=ROOT/'configs/bilin18_mlp_input_baseline_fresh_locked.json';frozen=json.loads(lock.read_text());vals=frozen['values'];keys=frozen['keys'];rows=[]
 for i in range(16):
  for f in [0,1]:
   for sk in [False,True]:
    for sv in [True]:
     k1=keys[i];k2=k1 if sk else keys[(i+5)%16];v1=vals[i];v2=v1 if sv else vals[(i+3)%16]
     for query in [0,1]:
      kq=[k1,k2][query];text=f'{k1} {v1} | {k2} {v2} | {kq}' if f==0 else f'{k1}: {v1}\n{k2}: {v2}\n{kq}:';rows.append({'id':f'h807-{f}-{i:02d}-{int(sk)}-{int(sv)}-{query}','item':i,'format':f,'same_key':sk,'same_value':sv,'query':query,'text':text,'values':[v1,v2],'contrast_values':[v1,'train'],'expected_index':None if sk and not sv else (1 if query==1 and not sv else 0)})
 spec=ART/f'{S}_manifest.json';spec.write_text(json.dumps(rows,indent=2)+'\n');cfg={'experiment':'H807','locked_config_sha256':sha(lock),'design':'Frozen bird vocabulary; same values, same/different keys, two queries; align four cells per item/format. Candidate diagnostic compares correct bird word with fixed train foil; full-vocabulary correctness also reported. No outcome exclusions.','manifest_sha256':sha(spec),'runner_sha256':sha(Path(__file__)),'model_sha256':sha(SNAPSHOT/'pytorch_model.bin')};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);cells={};excluded=[]
 for row in rows:
  enc=tok(row['text'],add_special_tokens=False,return_offsets_mapping=True);spans=[]
  if row['same_value']:spans=[m.span() for m in re.finditer(r'\b'+row['values'][0]+r'\b',row['text'])]
  else:spans=[re.search(r'\b'+v+r'\b',row['text']).span() for v in row['values']]
  pp=[]
  for a,b in spans:
   ix=[i for i,(c,d) in enumerate(enc['offset_mapping']) if c<b and d>a]
   if len(ix)==1:pp.append(ix[0])
  ids=[continuation_id(tok,row['text'],' '+v) for v in row['contrast_values']]
  if len(pp)!=2 or None in ids or len(set(ids))!=2:excluded.append({'id':row['id'],'reason':'single-token value/answer validity'});continue
  cells[row['id']]={**row,'ids':enc['input_ids'],'positions':pp,'answer_ids':ids}
 groups=defaultdict(list)
 for e in cells.values():groups[e['format'],e['item']].append(e)
 valid=[]
 for (f,i),ee in groups.items():
  if len(ee)!=4 or len({tuple(e['positions']) for e in ee})!=1 or len({len(e['ids']) for e in ee})!=1:excluded.append({'format':f,'item':i,'reason':'factorial length/position alignment'});continue
  valid.extend(ee)
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 attn=model.blocks[2].attn;original=attn.squared_attention;batch=[];saved={}
 def normcap(m,args):saved['u']=torch.stack([args[0][i,e['positions']] for i,e in enumerate(batch)]).cpu()
 def intercept(q,k,v,q2,k2):
  pp=torch.stack([torch.stack([q[i,e['positions'][1],5],k[i,e['positions'][0],5],q2[i,e['positions'][1],5],k2[i,e['positions'][0],5],v[i,e['positions'][0],5]]) for i,e in enumerate(batch)]);saved['factors']=pp.cpu();return original(q,k,v,q2,k2)
 h=attn.register_forward_pre_hook(normcap);attn.squared_attention=intercept;buckets=defaultdict(list)
 for e in valid:buckets[len(e['ids'])].append(e)
 records=[];captures={};forwards=0
 try:
  for ee in buckets.values():
   for lo in range(0,len(ee),12):
    batch=ee[lo:lo+12];out,_=run(model,torch.tensor([e['ids'] for e in batch],device='cuda'));forwards+=len(batch)
    for i,e in enumerate(batch):
     q,k,q2,k2,v=saved['factors'][i].double();s1=float(q@k/128);s2=float(q2@k2/128);u=saved['u'][i].double();p=out[i].cpu();expected=e['expected_index'];r={k:e[k] for k in ['id','item','format','same_key','same_value','query','expected_index']};r.update(score1=s1,score2=s2,pattern=s1*s2,input_cosine=float(torch.nn.functional.cosine_similarity(u[0],u[1],dim=0)),candidate_margin=float(p[e['answer_ids'][0]]-p[e['answer_ids'][1]]),candidate_correct=None if expected is None else int(p[e['answer_ids']].argmax())==expected,full_vocab_correct=None if expected is None else int(p.argmax())==e['answer_ids'][expected]);records.append(r);captures[e['id']]={'factors':saved['factors'][i],'normalized_input':saved['u'][i]}
 finally:h.remove();attn.squared_attention=original
 idx={(r['format'],r['item'],r['same_key'],r['same_value'],r['query']):r for r in records};future=[]
 for r in records:
  if r['query']==0:
   other=idx[r['format'],r['item'],r['same_key'],r['same_value'],1];future.extend(abs(r[m]-other[m]) for m in ['score1','score2','pattern','input_cosine'])
 audits={'hashes_unchanged':sha(spec)==cfg['manifest_sha256'] and sha(Path(__file__))==cfg['runner_sha256'] and sha(SNAPSHOT/'pytorch_model.bin')==cfg['model_sha256'],'future_query_source_coefficient_max_abs':max(future),'finite':all(math.isfinite(r['pattern']) for r in records),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(future)<1e-5
 torch.save(captures,ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'excluded':excluded,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits);print('excluded',excluded)
if __name__=='__main__':main()
