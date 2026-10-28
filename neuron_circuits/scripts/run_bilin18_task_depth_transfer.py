"""H719: full final-position residual transfer across depth."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch,numpy as np
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART,groups as answer_groups,measure
from run_bilin18_broad_task_competence import continuation_id
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_task_depth_transfer'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'old':ART/'bilin18_group_task_transfer.json','old_captures':ART/'bilin18_group_task_transfer_captures.pt','manifest':ART/'bilin18_group_task_transfer_manifest.json','early_swap':ART/'bilin18_task_span_controls.json','suite':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','tokenization':ROOT/'scripts/run_bilin18_broad_task_competence.py','model':SNAPSHOT/'pytorch_model.bin','runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py'};hashes={k:sha(p) for k,p in paths.items()};old=json.loads(paths['old'].read_text());rows=json.loads(paths['manifest'].read_text());tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={}
 for row in rows:
  if not old['competence'][row['family']]['discovery']['passed']:continue
  for v,text in enumerate(row['prompts']+[row['nuisance']]):
   gs=[[continuation_id(tok,text,a) for a in g] for g in answer_groups(row['family'],row['answers'])];gs=[[a for a in g if a is not None] for g in gs];assert all(gs);examples[row['id'],v]=dict(ids=tok.encode(text,add_special_tokens=False),groups=gs)
 cfg=dict(experiment='H719',hashes=hashes,design='All80competence-eligible H717pairs with3variants, noindividualfiltering. Capture last-position residual aftereachof18blocks. Replace only receiverlast-position residualwithdonor residual afteroneblock, keepingreceiverbus/otherpositions/inputpathway. Both taskdirections andnuisance0<-2. Verifyfinalblock donorlogit replay first; scanallotherlayers. Predeclared first discoverylayer with>=50%mean availabledonor-margin recovery andpositive task-minusabsolute-nuisance score, thenevaluate same layer ontransferwithoutreselection. Preservefullcurvesandfailed thresholds.',limitations='Whole-state sufficiency scan, notlocalcomputation ornecessity; overwrite bypasses earlierprocesses. Differentdonor/receiverprefixlengths allowed, allotherpositionsandbusreceiver. Final-layerreplay is architectural positivecontrol, notdiscoveredmechanism. Reusedtask suite; corpusreservedtestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 states={};native={};buckets=defaultdict(list);forwards=0
 for k,e in examples.items():buckets[len(e['ids'])].append(k)
 for length,keys in buckets.items():
  for st in range(0,len(keys),12):
   bb=keys[st:st+12];cap={};hooks=[]
   for layer in range(18):
    def get(m,args,out,layer=layer):cap[layer]=out[0][:,-1].cpu()
    hooks.append(model.blocks[layer].register_forward_hook(get))
   try:logits,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:
    for h in hooks:h.remove()
   forwards+=len(bb)
   for n,k in enumerate(bb):native[k]=logits[n].cpu();states[k]=torch.stack([cap[l][n] for l in range(18)])
 previous=torch.load(paths['old_captures'],weights_only=False);native_error=max(float((v-previous['native_logits'][k]).abs().max()) for k,v in native.items());state_error=max(float((v[1]-previous['states'][k]).norm()/previous['states'][k].norm()) for k,v in states.items());del previous;print('NATIVE',len(native),flush=True)
 swaps=[]
 for row in rows:
  if (row['id'],0) in examples:
   for rv,dv in [(0,1),(1,0),(0,2)]:swaps.append(dict(id=row['id'],family=row['family'],split=row['split'],receiver=rv,donor=dv))
 records=[];finalerrors=[];buckets=defaultdict(list)
 for r in swaps:buckets[len(examples[r['id'],r['receiver']]['ids'])].append(r)
 for layer in [17]+list(range(17)):
  for length,jobs in buckets.items():
   for st in range(0,len(jobs),12):
    bb=jobs[st:st+12];donorstate=torch.stack([states[r['id'],r['donor']][layer] for r in bb]).cuda()
    def edit(m,args,out):
     x,bus=out;x=x.clone();x[:,-1]=donorstate.to(x);return x,bus
    hook=model.blocks[layer].register_forward_hook(edit)
    try:logits,_=run(model,torch.tensor([examples[r['id'],r['receiver']]['ids'] for r in bb],device='cuda'))
    finally:hook.remove()
    forwards+=len(bb)
    for n,r in enumerate(bb):
     rk,dk=(r['id'],r['receiver']),(r['id'],r['donor']);gs=examples[rk]['groups'];base=measure(native[rk],gs);edited=measure(logits[n],gs);don=measure(native[dk],gs);change=edited['margin']-base['margin'];sign=1 if r['donor'] in [0,2] else -1;bp=native[rk].double().log_softmax(-1);ep=logits[n].cpu().double().log_softmax(-1);dp=native[dk].double().log_softmax(-1);records.append(dict(**r,layer=layer,margin_change=change,donor_directed_margin_change=sign*change,available_donor_margin=sign*(don['margin']-base['margin']),native_receiver_correct=base['top1'][r['receiver']],native_donor_correct=don['top1'][0 if r['donor']==2 else r['donor']],edited_donor_correct=edited['top1'][0 if r['donor']==2 else r['donor']],edited_receiver_correct=edited['top1'][r['receiver']],receiver_kl=float((bp.exp()*(bp-ep)).sum()),donor_kl=float((dp.exp()*(dp-ep)).sum()),relative_edit_norm=float((states[dk][layer]-states[rk][layer]).norm()/states[rk][layer].norm())))
     if layer==17:finalerrors.append(float((logits[n].cpu()-native[dk]).abs().max()))
  if layer==17:assert max(finalerrors)<1e-4;print('FINAL_REPLAY',max(finalerrors),flush=True)
  print('LAYER',layer,flush=True);(ART/f'{STEM}.partial.json').write_text(json.dumps(dict(status='running',records=records),indent=2))
 oldswap=json.loads(paths['early_swap'].read_text());oldindex={(r['id'],r['receiver'],r['donor']):r for r in oldswap['records'] if r['mode']=='full' and r['dose']=='natural'};earlyerr=max(abs(r['margin_change']-oldindex[r['id'],r['receiver'],r['donor']]['margin_change']) for r in records if r['layer']==1)
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['receiver_kl']) for r in records),native_logits_previous_max_abs=native_error,block1_previous_state_max_relative=state_error,block1_swap_margin_previous_max_abs=earlyerr,final_block_donor_logits_max_abs=max(finalerrors),whole_model_forwards=forwards,native_forwards=len(native),test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and native_error<1e-4 and state_error<1e-4 and earlyerr<1e-4;torch.save(dict(native_logits=native,states=states),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
