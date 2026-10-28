"""H676: fixed demonstration prompts with prospective competence rule."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
from run_bilin18_operation_control_panel import declarations as old_declarations
from run_bilin18_value_rule_fresh_panel import declarations as new_declarations
STEM='bilin18_demonstration_competence'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);old=old_declarations();new=new_declarations();rows=[]
 for family in ['plural','antonym','capital','successor','addition','agreement']:
  demos=[x for x in old if x['family']==family and x['item']<4];tests=[dict(x,item=n) for n,x in enumerate([x for x in old if x['family']==family and x['item']>=4]+[x for x in new if x['family']==family])];assert len(demos)==4 and len(tests)==8 and not {x['input'] for x in demos}&{x['input'] for x in tests}
  for control in ['task','copy']:
   key='answer' if control=='task' else 'copy_answer';instruction=f"Give {demos[0]['operation']} for each input." if control=='task' else 'Repeat the first word or number of each input unchanged.'
   for form in ['arrow','labeled']:
    def example(x,answer):return f"{x['input']} ->{(' '+answer) if answer is not None else ''}" if form=='arrow' else f"Input: {x['input']}\nAnswer:{(' '+answer) if answer is not None else ''}"
    prefix=instruction+'\n'+'\n'.join(example(x,x[key]) for x in demos)+'\n'
    for t in tests:
     prompt=prefix+example(t,None);target=tok.encode(' '+t[key],add_special_tokens=False);assert len(target)==1;rows.append(dict(**t,id=f"{family}-{t['item']}-{control}-{form}",control=control,form=form,expected=t[key],answer_id=target[0],prompt=prompt,ids=tok.encode(prompt,add_special_tokens=False),split='competence_screen'))
 assert len(rows)==192 and len({r['prompt'] for r in rows})==192
 manifest=ART/f'{STEM}_manifest.json';manifest.write_text(json.dumps(rows,indent=2));paths={'manifest':manifest,'runner':Path(__file__),'old_declarations':ROOT/'scripts/run_bilin18_operation_control_panel.py','new_declarations':ROOT/'scripts/run_bilin18_value_rule_fresh_panel.py'};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};cfg=dict(experiment='H676',hashes=hashes,design='6 families x8 query items x2 controls x2 formats, four demonstrations each. Full-vocabulary next-token accuracy. Qualify family/form only if task>=6/8 and copy>=6/8. All cells retained. Frozen manifest/config before model loading.',limitations='New prompts, reused vocabulary and query items. Competence screening, not independent intervention validation. Agreement copying always The; its control does not establish item-sensitive copying. One fixed demonstration order; answer length single token only.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 buckets=defaultdict(list)
 for i,r in enumerate(rows):buckets[len(r['ids'])].append(i)
 logits=torch.empty(len(rows),model.lm_head.weight.shape[0]);caches={};forwards=0
 for inds in buckets.values():
  for st in range(0,len(inds),8):
   bb=inds[st:st+8];z,cc=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'),capture=True);logits[bb]=z.cpu()
   for n,i in enumerate(bb):caches[i]={key:value[n].clone() for key,value in cc.items()}
   forwards+=len(bb)
 records=[]
 for i,r in enumerate(rows):
  lp=logits[i].double().log_softmax(-1);top=int(lp.argmax());records.append(dict(id=r['id'],family=r['family'],form=r['form'],control=r['control'],item=r['item'],correct=top==r['answer_id'],target_ce=float(-lp[r['answer_id']]),prediction=tok.decode([top]),expected=r['expected'],target_rank=int((lp>lp[r['answer_id']]).sum())+1))
 cells=[]
 for f in ['plural','antonym','capital','successor','addition','agreement']:
  for form in ['arrow','labeled']:
   counts={c:sum(x['correct'] for x in records if x['family']==f and x['form']==form and x['control']==c) for c in ['task','copy']};cells.append(dict(family=f,form=form,counts=counts,qualified=min(counts.values())>=6))
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},whole_model_forwards=forwards,prompts=192,finite=bool(torch.isfinite(logits).all()),demo_query_inputs_disjoint=True)
 torch.save(dict(native_logits=logits,caches=caches),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,cells=cells,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(cells);print(audits)
if __name__=='__main__':main()
