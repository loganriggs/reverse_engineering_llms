"""H789 frozen fresh lexical/format panel and native L1/L4 capture."""
import json,time
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_broad_task_competence import continuation_id
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
S='bilin18_native_scale_replication'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');keys='oak pine elm ash birch cedar maple willow beech fir spruce palm moss fern reed ivy'.split();values='table chair desk bed sofa shelf lamp couch plate cup bowl fork spoon knife glass pot'.split();names='Victor Olivia Nathan Sophia Ethan Chloe Lucas Mia Oscar Nora Leo Zoe Felix Iris Hugo Eva Adrian Clara Simon Julia Arthur Leah Miles Stella Owen Violet Caleb Hazel Jasper Ruby Isaac Elena'.split();rows=[]
 for i in range(16):
  key=keys[i];v,w=values[i],values[(i+5)%16];a,b=names[2*i:2*i+2]
  for f in [0,1]:
   copy=lambda k,x: f'{k} {x} | {k} {x} | {k}' if f==0 else f'{k}: {x}\n{k}: {x}\n{k}:'
   binding=lambda giver,obj: f'{a} and {b} met at the station. {giver} passed a {obj} to' if f==0 else f'{a} met {b} at the cafe. Later, {giver} offered a {obj} to'
   for task,prompts,answers,nuisance in [('sequence_copy',[copy(key,x) for x in [v,w]],[v,w],copy(keys[(i+7)%16],v)),('entity_binding',[binding(g,'parcel' if f==0 else 'ticket') for g in [a,b]],[b,a],binding(a,'suitcase' if f==0 else 'map'))]:rows.append({'id':f'h789-{task}-f{f}-{i:02d}','family':f'{task}_f{f}','task':task,'format':f,'item':i,'prompts':prompts,'answers':answers,'nuisance':nuisance})
 spec=ART/f'{S}_manifest.json';spec.write_text(json.dumps(rows,indent=2)+'\n');plan={'primary':'L4 sequence copying sign-sensitive donor-directed effect positive and greater than random at fractions .25/.5/1; true versus shuffled tests item specificity. Native net swaps separately reported.','comparators':['L1','entity binding'],'gate':'Each task/format cell: >=4 token-valid pairs, >=.75 paired endpoint choice, >=.5 intended full-vocabulary top1. No individual correctness filtering.','analysis':'Report every format separately. Bootstrap item clusters within cell; pooled cross-format inference must cluster shared item index. No selection of favorable fractions or formats.','novelty':'Fresh lists and prompt constructions relative to H785 panel, not guaranteed absent from all historical experiments or model pretraining. Lexical categories and formats both change.','manifest_sha256':sha(spec)};(ART/f'{S}_preregistration.json').write_text(json.dumps(plan,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={};excluded=[]
 for row in rows:
  ee=[]
  for variant,text in enumerate(row['prompts']+[row['nuisance']]):
   gs=[[continuation_id(tok,text,' '+answer)] for answer in row['answers']]
   if any(g[0] is None for g in gs) or gs[0]==gs[1]:break
   ee.append({'ids':tok.encode(text,add_special_tokens=False),'groups':gs,'family':row['family'],'task':row['task'],'format':row['format'],'item':row['item']})
  if len(ee)!=3:excluded.append(row['id']);continue
  for variant,e in enumerate(ee):examples[row['id'],variant]=e
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 inputs={};outputs={};native={};saved={};handles=[];buckets=defaultdict(list)
 for l in [1,4]:
  def pre(m,args,l=l):saved[l,'input']=args[0][:,-1].cpu()
  def post(m,args,out,l=l):saved[l,'output']=out[:,-1].cpu()
  handles.extend([model.blocks[l].mlp.register_forward_pre_hook(pre),model.blocks[l].mlp.register_forward_hook(post)])
 for k,e in examples.items():buckets[len(e['ids'])].append(k)
 try:
  for kk in buckets.values():
   for lo in range(0,len(kk),12):
    bb=kk[lo:lo+12];out,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
    for i,k in enumerate(bb):native[k]=out[i].cpu();inputs[k]={l:saved[l,'input'][i] for l in [1,4]};outputs[k]={l:saved[l,'output'][i] for l in [1,4]}
 finally:
  for h in handles:h.remove()
 competence=[]
 for fam in sorted({r['family'] for r in rows}):
  ids=[r['id'] for r in rows if r['family']==fam and (r['id'],0) in examples];both=[];top=[];nuis=[]
  for rid in ids:
   mm=[measure(native[rid,v],examples[rid,v]['groups']) for v in [0,1,2]];both.append(mm[0]['margin']>0 and mm[1]['margin']<0);top.extend([mm[v]['top1'][v] for v in [0,1]]);nuis.append(mm[2]['margin']>0)
  avg=lambda z:sum(z)/len(z) if z else 0.;competence.append({'family':fam,'pairs':len(ids),'both_endpoint_choice':avg(both),'full_vocab_top1':avg(top),'nuisance_choice':avg(nuis),'pass':len(ids)>=4 and avg(both)>=.75 and avg(top)>=.5})
 assert sha(spec)==plan['manifest_sha256'];torch.save({'inputs':inputs,'outputs':outputs,'native':native,'examples':examples},ART/f'{S}_captures.pt');(ART/f'{S}_native.json').write_text(json.dumps({'competence':competence,'excluded':excluded,'native_forwards':len(examples),'preregistration':plan},indent=2)+'\n');print(json.dumps(competence,indent=2));print('excluded',excluded)
if __name__=='__main__':main()
