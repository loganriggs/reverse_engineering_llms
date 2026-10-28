"""H678: paired instruction coefficient swaps with fixed receiver value."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
STEM='bilin18_competent_gate_swap'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_demonstration_competence_manifest.json','native':ART/'bilin18_demonstration_competence_captures.pt','source':ART/'bilin18_demo_source_removal.json','writes':ART/'bilin18_demo_source_removal_captures.pt','roles':ART/'bilin18_demo_source_removal_roles.json','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['native'],weights_only=False);src=json.loads(paths['source'].read_text());writes=torch.load(paths['writes'],weights_only=False);roles=json.loads(paths['roles'].read_text());g={x['id']:x['query_head4_coefficients'] for x in src['gates']};cells={('plural','arrow'),('plural','labeled'),('successor','arrow')};held=[i for i,r in enumerate(rows) if (r['family'],r['form']) in cells];assert len(held)==48 and all(len(roles[i]['query_positions'])==len(g[rows[i]['id']])==1 for i in held);lookup={(r['family'],r['form'],r['control'],r['item']):i for i,r in enumerate(rows)};donors={};coef=torch.tensor([g[rows[i]['id']][0] for i in held],dtype=torch.float64)
 for mode in ['opposite','same_instruction_other_item']:
  dd=[]
  for i in held:
   r=rows[i];c=('copy' if r['control']=='task' else 'task') if mode=='opposite' else r['control'];item=r['item'] if mode=='opposite' else (r['item']+1)%8;dd.append(lookup[r['family'],r['form'],c,item])
  donors[mode]=dd
 cfg=dict(experiment='H678',hashes=hashes,design='48 prompts,3 prequalified single-token cells. Native replay; replace queryhead4 scalar with paired opposite-instruction coefficient or cyclic nextitem same-instruction coefficient. Receiver effective value fixed; full downstream computation. All8items retained.',limitations='Opposite prompts differ in instruction and demonstrations, not instruction text alone. Same-instruction donor is one fixed cyclic control, not normmatched or a null distribution. Local coefficient swap not sufficient full-task interchange.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 att=model.blocks[14].attn;W=att.c_proj.weight[:,512:640].double();V=torch.empty(48,1152,dtype=torch.float64);buckets=defaultdict(list)
 for k,i in enumerate(held):buckets[len(rows[i]['ids'])].append(k)
 outputs={};forwards=0
 for mode in ['baseline','opposite','same_instruction_other_item']:
  zall=torch.empty(48,model.lm_head.weight.shape[0])
  dc=torch.zeros(48,dtype=torch.float64) if mode=='baseline' else torch.tensor([g[rows[i]['id']][0] for i in donors[mode]],dtype=torch.float64)-coef
  for inds in buckets.values():
   for st in range(0,len(inds),8):
    bb=inds[st:st+8]
    if mode=='baseline':
     def inspect(m,args,out):
      y,bus=args;B,T,_=y.shape;values=(1-att.lamb)*att.c_v(y).view(B,T,9,128)+att.lamb*bus.view(B,T,9,128)
      for n,k in enumerate(bb):V[k]=(values[n,roles[held[k]]['query_positions'][0],4].double()@W.T).cpu()
     hook=att.register_forward_hook(inspect)
    else:
     patch=dc[bb,None]*V[bb]
     def edit(m,args,out):
      y,b=out;y=y.clone();y[:,-1]+=patch.to(y);return y,b
     hook=att.register_forward_hook(edit)
    try:z,_=run(model,torch.tensor([rows[held[k]]['ids'] for k in bb],device='cuda'));zall[bb]=z.cpu()
    finally:hook.remove()
    forwards+=len(bb)
  outputs[mode]=zall;print('COMPLETE',mode,flush=True)
 native=cap['native_logits'][held].double();lp0=native.log_softmax(-1);p0=lp0.exp();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 def token(s):
  ids=tok.encode(' '+s,add_special_tokens=False);assert len(ids)==1;return ids[0]
 records=[]
 for mode,z in outputs.items():
  lp=z.double().log_softmax(-1)
  for k,i in enumerate(held):
   r=rows[i];ans,cur=token(r['answer']),token(r['copy_answer']);tid=r['answer_id'];top=int(z[k].argmax());records.append(dict(id=r['id'],family=r['family'],form=r['form'],control=r['control'],item=r['item'],mode=mode,coefficient=float(coef[k]),donor_coefficient=float(coef[k]) if mode=='baseline' else g[rows[donors[mode][k]]['id']][0],target_ce_delta=float(lp0[k,tid]-lp[k,tid]),task_minus_copy_delta=float(z[k,ans]-z[k,cur]-native[k,ans]+native[k,cur]),native_kl=float((p0[k]*(lp0[k]-lp[k])).sum()),top1_correct=top==tid,task_top1=top==ans,copy_top1=top==cur))
 replay=float((outputs['baseline'].double()-native).abs().max());truth=writes['vectors']['query_head4'][held];wr=float(((coef[:,None]*V-truth).norm(dim=-1)/truth.norm(dim=-1).clamp_min(1e-12)).max());assert replay<1e-3 and wr<1e-5
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},native_replay_max_abs=replay,query_write_replay_max_relative=wr,whole_model_forwards=forwards,finite=all(bool(torch.isfinite(v).all()) for v in outputs.values()));torch.save(dict(logits=outputs,heldout_indices=held,receiver_values=V,donors=donors),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits)
if __name__=='__main__':main()
