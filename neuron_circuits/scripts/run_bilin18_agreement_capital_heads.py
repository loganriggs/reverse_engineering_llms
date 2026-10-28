"""H721: live native-head transfer for agreement and capital recall."""
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
STEM='bilin18_agreement_capital_heads'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'manifest':ART/'bilin18_group_task_transfer_manifest.json','components':ART/'bilin18_task_transition_components.json','prior_readers':ART/'BILIN18_BUS_READER_DECOMPOSITION.md','prior_routes':ART/'BILIN18_TASK_ROUTE_LOCALIZATION.md','suite':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','tokenization':ROOT/'scripts/run_bilin18_broad_task_competence.py','model_source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py'),'model':SNAPSHOT/'pytorch_model.bin','runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py'};hashes={k:sha(p) for k,p in paths.items()};selected={'agreement':11,'capital_recall':13};rows=json.loads(paths['manifest'].read_text());tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={}
 for row in rows:
  if row['family'] not in selected:continue
  for v,text in enumerate(row['prompts']+[row['nuisance']]):
   gs=[[continuation_id(tok,text,a) for a in g] for g in answer_groups(row['family'],row['answers'])];gs=[[a for a in g if a is not None] for g in gs];assert all(gs);examples[row['id'],v]=dict(ids=tok.encode(text,add_special_tokens=False),groups=gs,layer=selected[row['family']])
 cfg=dict(experiment='H721',hashes=hashes,selected_layers=selected,design='All9nativeheads atagreementL11andcapitalL13,32paireditems/96variants. Capturec_projinput lastposition9x128andprojecteachheadseparately; auditheadsumagainstnativeattentionwrite. Replaceonehead128Dpreprojectionvaluewithdonor, orallheads. Receiverotherpositions/busunchanged; recomputeMLPanddownstream normally. Both taskdirections+nuisance0<-2. Selecthead by discovery mean taskdonor-margin gain minusabsolute nuisancechange; evaluatefrozenheadontransfer; retainallheads. BindingL14H4andcopyL8H3priorresultsreviewedratherthanrepeated.',limitations='Naturalheadswapsnotnormmatched. Onehead isarchitecturalunitnotnecessarilyonefeatureoroperation. Reusedsuite,8transferpairsperfamily; no universalheadlabel ornecessity claim. Originalcorpusreservedtestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 native={};heads={};writes={};buckets=defaultdict(list);forwards=0;headsum=[]
 for k,e in examples.items():buckets[e['layer'],len(e['ids'])].append(k)
 for (layer,length),keys in buckets.items():
  block=model.blocks[layer];assert block.attn.c_proj.bias is None
  for st in range(0,len(keys),12):
   bb=keys[st:st+12];cap={}
   def pre(m,args):cap['h']=args[0][:,-1].reshape(-1,9,128).cpu()
   def attn(m,args,out):cap['a']=out[0][:,-1].cpu()
   hs=[block.attn.c_proj.register_forward_pre_hook(pre),block.attn.register_forward_hook(attn)]
   try:logits,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:
    for h in hs:h.remove()
   forwards+=len(bb);w=torch.einsum('bhd,ohd->bho',cap['h'].double(),block.attn.c_proj.weight.cpu().double().reshape(1152,9,128));headsum.extend(((w.sum(1)-cap['a'].double()).norm(dim=1)/cap['a'].double().norm(dim=1)).tolist())
   for n,k in enumerate(bb):native[k]=logits[n].cpu();heads[k]=cap['h'][n];writes[k]=w[n].float()
 assert max(headsum)<1e-5;records=[]
 for head in list(range(9))+[-1]:
  jobs=defaultdict(list)
  for row in rows:
   if row['family'] not in selected:continue
   for rv,dv in [(0,1),(1,0),(0,2)]:jobs[selected[row['family']],len(examples[row['id'],rv]['ids'])].append(dict(id=row['id'],family=row['family'],split=row['split'],receiver=rv,donor=dv,layer=selected[row['family']],head=head))
  for (layer,length),jj in jobs.items():
   for st in range(0,len(jj),12):
    bb=jj[st:st+12];don=torch.stack([heads[r['id'],r['donor']] for r in bb]).cuda()
    def edit(m,args):
     y=args[0].clone();last=y[:,-1].reshape(-1,9,128)
     if head<0:last[:]=don.to(last)
     else:last[:,head]=don[:,head].to(last)
     return (y,)
    hook=model.blocks[layer].attn.c_proj.register_forward_pre_hook(edit)
    try:logits,_=run(model,torch.tensor([examples[r['id'],r['receiver']]['ids'] for r in bb],device='cuda'))
    finally:hook.remove()
    forwards+=len(bb)
    for n,r in enumerate(bb):
     rk,dk=(r['id'],r['receiver']),(r['id'],r['donor']);gs=examples[rk]['groups'];base=measure(native[rk],gs);ed=measure(logits[n],gs);dn=measure(native[dk],gs);change=ed['margin']-base['margin'];sign=1 if r['donor'] in [0,2] else -1;bp=native[rk].double().log_softmax(-1);ep=logits[n].cpu().double().log_softmax(-1);delta=writes[dk]-writes[rk];delta=delta.sum(0) if head<0 else delta[head];records.append(dict(**r,margin_change=change,donor_directed_margin_change=sign*change,available_donor_margin=sign*(dn['margin']-base['margin']),edited_donor_correct=ed['top1'][0 if r['donor']==2 else r['donor']],receiver_kl=float((bp.exp()*(bp-ep)).sum()),head_write_edit_norm=float(delta.norm())))
  print('HEAD',head,flush=True);(ART/f'{STEM}.partial.json').write_text(json.dumps(dict(status='running',records=records),indent=2))
 old=json.loads(paths['components'].read_text());idx={(r['id'],r['receiver'],r['donor']):r for r in old['records'] if r['mode']=='live_attention'};replay=max(abs(r['margin_change']-idx[r['id'],r['receiver'],r['donor']]['margin_change']) for r in records if r['head']==-1);audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['receiver_kl']) for r in records),head_sum_native_attention_max_relative=max(headsum),allhead_live_attention_margin_replay_max_abs=replay,whole_model_forwards=forwards,native_forwards=len(native),test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and replay<1e-4;torch.save(dict(native_logits=native,head_coordinates=heads,head_writes=writes),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
