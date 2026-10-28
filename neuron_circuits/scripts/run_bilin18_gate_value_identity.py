"""H680: exhaustive coefficient x value-identity factorial on competent month receivers."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
STEM='bilin18_gate_value_identity'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_demonstration_competence_manifest.json','source':ART/'bilin18_competent_gate_swap.json','capture':ART/'bilin18_competent_gate_swap_captures.pt','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};rows=json.loads(paths['manifest'].read_text());src=json.loads(paths['source'].read_text());cap=torch.load(paths['capture'],weights_only=False);hh=cap['heldout_indices'];selected=[k for k,i in enumerate(hh) if rows[i]['family']=='successor' and rows[i]['control']=='copy'];assert len(selected)==8;rr=[rows[hh[k]] for k in selected];V=cap['receiver_values'][selected];baseline={r['id']:r for r in src['records'] if r['mode']=='baseline'};swapped={r['id']:r for r in src['records'] if r['mode']=='opposite'};cc=torch.tensor([baseline[r['id']]['coefficient'] for r in rr],dtype=torch.float64);ct=torch.tensor([swapped[r['id']]['donor_coefficient'] for r in rr],dtype=torch.float64);conditions=[dict(receiver=i,donor=i,gate='baseline',scale='native') for i in range(8)]
 for scale in ['native','norm_matched']:
  for gate in ['copy','task']:
   for i in range(8):
    for d in range(8):conditions.append(dict(receiver=i,donor=d,gate=gate,scale=scale))
 cfg=dict(experiment='H680',hashes=hashes,design='8 original competent month-copy receivers, exhaustive8x8 value identity crossing with receiver copy or matched task coefficient; native and receiver-norm-matched donor values. Replace full native queryhead4 write with newcoefficient*donorvalue.8 zero-edit baselines,264 forwards. No selection by outcome.',limitations='Original repeatedly inspected prompts, counterfactual donor values may be off manifold. This is one component replacement; all other receiver content unchanged. Donor/receiver current/next labels sometimes overlap and are not mutually exclusive metrics.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 patches=[]
 for c in conditions:
  i,d=c['receiver'],c['donor'];v=V[d]*(V[i].norm()/V[d].norm() if c['scale']=='norm_matched' else 1);newc=ct[i] if c['gate']=='task' else cc[i];patches.append(torch.zeros_like(v) if c['gate']=='baseline' else newc*v-cc[i]*V[i])
 patches=torch.stack(patches);buckets=defaultdict(list)
 for k,c in enumerate(conditions):buckets[len(rr[c['receiver']]['ids'])].append(k)
 outputs=torch.empty(len(conditions),model.lm_head.weight.shape[0])
 for inds in buckets.values():
  for st in range(0,len(inds),8):
   bb=inds[st:st+8];patch=patches[bb]
   def edit(m,args,out):
    y,b=out;y=y.clone();y[:,-1]+=patch.to(y);return y,b
   hook=model.blocks[14].attn.register_forward_hook(edit)
   try:z,_=run(model,torch.tensor([rr[conditions[k]['receiver']]['ids'] for k in bb],device='cuda'));outputs[bb]=z.cpu()
   finally:hook.remove()
 native=cap['logits']['baseline'][selected].double();lp0=native.log_softmax(-1);p0=lp0.exp();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 def token(s):
  ids=tok.encode(' '+s,add_special_tokens=False);assert len(ids)==1;return ids[0]
 records=[];diagonal=[]
 for k,c in enumerate(conditions):
  i,d=c['receiver'],c['donor'];lp=outputs[k].double().log_softmax(-1);top=int(outputs[k].argmax());ids={'receiver_current':token(rr[i]['input']),'receiver_next':token(rr[i]['answer']),'donor_current':token(rr[d]['input']),'donor_next':token(rr[d]['answer'])};entry=dict(**c,receiver_input=rr[i]['input'],donor_input=rr[d]['input'],prediction=tok.decode([top]),native_kl=float((p0[i]*(lp0[i]-lp)).sum()),receiver_copy_ce_delta=float(lp0[i,ids['receiver_current']]-lp[ids['receiver_current']]),receiver_next_minus_current_delta=float(outputs[k,ids['receiver_next']]-outputs[k,ids['receiver_current']]-native[i,ids['receiver_next']]+native[i,ids['receiver_current']]))
  for name,tid in ids.items():entry[name+'_top1']=top==tid;entry[name+'_logit_delta']=float(outputs[k,tid]-native[i,tid])
  records.append(entry)
  if i==d:
   truth=cap['logits']['opposite' if c['gate']=='task' else 'baseline'][selected[i]];diagonal.append(float((outputs[k]-truth).abs().max()))
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},diagonal_reference_replay_max_abs=max(diagonal),whole_model_forwards=len(conditions),finite=bool(torch.isfinite(outputs).all()));assert max(diagonal)<1e-3;torch.save(dict(logits=outputs,conditions=conditions),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits)
if __name__=='__main__':main()
