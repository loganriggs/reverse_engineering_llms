"""H675: finite removal of native input-position writes, full downstream replay."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
STEM='bilin18_native_input_removal'
@torch.no_grad()
def main():
 torch.set_num_threads(6);t=time.time();paths={'manifest':ART/'bilin18_operation_paraphrase_panel_manifest.json','native':ART/'bilin18_operation_paraphrase_panel_captures.pt','writes':ART/'bilin18_native_input_head_balance_captures.pt','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['native'],weights_only=False);writes=torch.load(paths['writes'],weights_only=False);held=writes['heldout_indices'];assert writes['component_order']==['local','bus'];V=writes['vectors'].sum(1);assert V.shape==(128,9,1152)
 cfg=dict(experiment='H675',hashes=hashes,design='128 existing contexts; remove0,half,or all of native layer14 input-position contribution from head4 or all9heads. Full downstream computation, fixed vectors including local and carried sources. Measure native KL, target CE and full-vocabulary correctness separately.',limitations='Activation removal may be off manifold; input position only, not whole-head ablation. Retrospective16 unique inputs. Native model often fails these task prompts.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 def token(v):
  ids=tok.encode(' '+v,add_special_tokens=False);assert len(ids)==1;return ids[0]
 buckets=defaultdict(list)
 for k,i in enumerate(held):buckets[len(rows[i]['ids'])].append(k)
 outputs={};records=[];forwards=0;native=cap['native_logits'][held].double();lp0=native.log_softmax(-1);p0=lp0.exp()
 for mode,strength in [('none',0.),('head4',.5),('head4',1.),('all',.5),('all',1.)]:
  zall=torch.empty_like(native,dtype=torch.float32);vec=V[:,4] if mode=='head4' else V.sum(1)
  for inds in buckets.values():
   for start in range(0,len(inds),16):
    bb=inds[start:start+16];hooks=[];patch=-strength*vec[bb]
    if strength:
     def edit(m,args,out):
      y,b=out;y=y.clone();y[:,-1]+=patch.to(y);return y,b
     hooks.append(model.blocks[14].attn.register_forward_hook(edit))
    try:z,_=run(model,torch.tensor([rows[held[k]]['ids'] for k in bb],device='cuda'));zall[bb]=z.cpu()
    finally:
     for h in hooks:h.remove()
    forwards+=len(bb)
  outputs[mode,strength]=zall;lp=zall.double().log_softmax(-1)
  for k,i in enumerate(held):
   r=rows[i];cur=token(r['input']);ans=token(r['answer']);target=r['answer_id'];records.append(dict(id=r['id'],family=r['family'],control=r['control'],form=r['form'],item=r['item'],mode=mode,strength=strength,current_delta=float(zall[k,cur]-native[k,cur]),answer_delta=float(zall[k,ans]-native[k,ans]),answer_minus_current_delta=float(zall[k,ans]-native[k,ans]-zall[k,cur]+native[k,cur]),native_kl=float((p0[k]*(lp0[k]-lp[k])).sum()),target_ce=float(-lp[k,target]),target_ce_delta=float(lp0[k,target]-lp[k,target]),top1_correct=int(zall[k].argmax())==target,native_top1_correct=int(native[k].argmax())==target))
  print('COMPLETE',mode,strength,flush=True)
 replay=float((outputs['none',0.].double()-native).abs().max());assert replay<1e-3
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},whole_model_forwards=forwards,native_replay_max_abs=replay,finite=all(torch.isfinite(z).all().item() for z in outputs.values()));torch.save(dict(logits=outputs,heldout_indices=held),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-t),indent=2));print(audits)
if __name__=='__main__':main()
