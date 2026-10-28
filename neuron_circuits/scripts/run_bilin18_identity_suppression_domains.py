"""H669: signed finite input-value effects on current/next/previous readouts."""
import json,hashlib,time
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
STEM='bilin18_identity_suppression_domains'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_operation_paraphrase_panel_manifest.json','capture':ART/'bilin18_operation_paraphrase_panel_captures.pt','factors':ART/'bilin18_head4_value_conditioned_captures.pt','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['capture'],weights_only=False);factors=torch.load(paths['factors'],weights_only=False);held=[i for i,r in enumerate(rows) if r['split']=='transfer' and r['family'] in ['plural','antonym','capital','successor'] and r['control'] in ['task','copy']];assert len(held)==128;positions={i:[k for k,v in enumerate(factors['token_metadata'][i]['labels']) if v==2] for i in held};assert all(len(v)==1 for v in positions.values());V=torch.stack([factors['projected_values'][i][positions[i][0]] for i in held]);g=torch.Generator().manual_seed(669);q=torch.randn(1152,generator=g,dtype=torch.float64);q/=q.norm();random=V.norm(dim=-1,keepdim=True)*q[None];alphas=[-.1,-.05,-.025,0.,.025,.05,.1];conditions=[('input',a) for a in alphas]+[('random',-.05),('random',.05)]
 cfg=dict(experiment='H670',hashes=hashes,design='128 heldout single-token-input copy/task prompts: plural nouns, antonym adjectives, capitals countries, succession months. Same fixed signed coefficient probes and seed669 norm-matched random control as H669. Read current input, task answer and all seven other input identities in each8-item family. Full downstream computation; no fitting.',limitations='Existing repeatedly inspected panel, not new blind replication. Single random control direction, limited vocabulary. Finite effect of one head value direction, not whole-head necessity or general semantic algorithm.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);ids={};otherids={}
 def token(x):
  tt=tok.encode(' '+x,add_special_tokens=False);assert len(tt)==1;return tt[0]
 for i in held:
  ids[i]={'current':token(rows[i]['input']),'answer':token(rows[i]['answer'])};otherids[i]=sorted({token(r['input']) for r in rows if r['family']==rows[i]['family'] and r['input']!=rows[i]['input']});assert len(otherids[i])==7
 buckets=defaultdict(list)
 for k,i in enumerate(held):buckets[len(rows[i]['ids'])].append(k)
 outputs={};records=[];forwards=0
 for kind,alpha in conditions:
  zall=torch.empty(len(held),model.lm_head.weight.shape[0]);vectors=V if kind=='input' else random
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];patch=alpha*vectors[bb];hooks=[]
    if alpha:
     def edit(m,args,out):y,b=out;y=y.clone();y[:,-1]+=patch.to(y);return y,b
     hooks.append(model.blocks[14].attn.register_forward_hook(edit))
    try:z,_=run(model,torch.tensor([rows[held[k]]['ids'] for k in bb],device='cuda'));zall[bb]=z.cpu()
    finally:
     for h in hooks:h.remove()
    forwards+=len(bb)
  outputs[kind,alpha]=zall
  for k,i in enumerate(held):
   r=rows[i];lp=zall[k].double().log_softmax(-1);record=dict(id=r['id'],family=r['family'],control=r['control'],form=r['form'],item=r['item'],kind=kind,alpha=alpha,value_norm=float(V[k].norm()),target_ce=float(-lp[r['answer_id']]),top1_correct=int(zall[k].argmax())==r['answer_id'])
   for label,tid in ids[i].items():record[label+'_logit']=float(zall[k,tid]);record[label+'_probability']=float(lp[tid].exp())
   records.append(record)
  print('COMPLETE',kind,alpha,flush=True)
 effects=[]
 for kind,amps in [('input',[.025,.05,.1]),('random',[.05])]:
  for amp in amps:
   plus,minus=outputs[kind,amp],outputs[kind,-amp]
   for k,i in enumerate(held):
    r=rows[i];slopes={label:float((plus[k,tid]-minus[k,tid])/(2*amp)) for label,tid in ids[i].items()};effects.append(dict(id=r['id'],family=r['family'],control=r['control'],form=r['form'],item=r['item'],kind=kind,amplitude=amp,slopes=slopes,answer_minus_current=slopes['answer']-slopes['current'],other_input_slope=float(((plus[k,otherids[i]]-minus[k,otherids[i]])/(2*amp)).mean()),other_minus_current=float(((plus[k,otherids[i]]-minus[k,otherids[i]])/(2*amp)).mean())-slopes['current']))
 replay=float((outputs['input',0.]-cap['native_logits'][held]).abs().max());assert replay<1e-3;audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},whole_model_forwards=forwards,heldout_prompts=len(held),native_replay_max_abs=replay,finite=all(np.isfinite(r['answer_minus_current']) and np.isfinite(r['other_minus_current']) for r in effects));torch.save(dict(logits=outputs,heldout_indices=held,random_unit_direction=q),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,effects=effects,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits)
if __name__=='__main__':main()
