"""H731: functional validation of frozen overlapping supports and rewired controls."""
import json,time,hashlib
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
from run_bilin18_verified_head_sparse import SignedBSF
from run_bilin18_learned_group_memberships import select_pairs,union_pairs
STEM='bilin18_learned_group_function'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_verified_head_pca_data.pt','fits':ART/'bilin18_verified_head_write_pca_fits.pt','models':ART/'bilin18_verified_head_sparse_captures.pt','implementation':ROOT/'scripts/run_bilin18_verified_head_sparse.py','heads':ART/'bilin18_agreement_capital_heads_captures.pt','head_results':ART/'bilin18_agreement_capital_heads.json','manifest':ART/'bilin18_group_task_transfer_manifest.json','suite':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','tokenization':ROOT/'scripts/run_bilin18_broad_task_competence.py','model':SNAPSHOT/'pytorch_model.bin','memberships':ART/'bilin18_learned_group_memberships_captures.pt','group_implementation':ROOT/'scripts/run_bilin18_learned_group_memberships.py','runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py'};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);fits=torch.load(paths['fits'],weights_only=False)['fits'];trained=torch.load(paths['models'],weights_only=False)['states'];members=torch.load(paths['memberships'],weights_only=False)['memberships'];old=torch.load(paths['heads'],weights_only=False);oldj=json.loads(paths['head_results'].read_text());rows=json.loads(paths['manifest'].read_text());sites={'agreement':(11,3),'capital_recall':(13,0)};cfg=dict(experiment='H731',hashes=hashes,design='All frozen H730 groups: 3dictionaryseeds, raw/random initializations and trained versions, each learned version with3degree-preservingrewires. ScalarTop8 and identity controls. Exactunion greedy plus4rounds alternatingbestresponse gates for allgroupconditions, no parameter updates. Rewired nulls preserve membership degrees but are only3 samples per fit. FrozenH721tasks/nuisance; delta=F(hdonor)-F(hreceiver), addtonative receiverhead, downstreamlive. Separatelynativeheadreplacement F(h) ontheexact32H722validationdocs atposition63. Measuretaskmargin/specificity, outputKL andobservedCE separately. No model/seedselectionbytaskresults. Identitycontrolsreplayfullheadswapsandnativeoutputs.',limitations='Single-head reconstruction andpartialtasktransfer, notfullmodelreplacement oraccuracy. Reusedpanels; 8transferpairs/family,32naturalcontexts. At most8unique activeatoms with2groupgates; scalarTop8 selects8freely. Group memberships and gates add structure/capacity. Reused functional panels are not fresh confirmatory data. Originalcorpusreservedtestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={}
 for row in rows:
  if row['family'] not in sites:continue
  for v,text in enumerate(row['prompts']+[row['nuisance']]):
   gs=[[continuation_id(tok,text,a) for a in g] for g in answer_groups(row['family'],row['answers'])];gs=[[a for a in g if a is not None] for g in gs];examples[row['id'],v]=dict(ids=tok.encode(text,add_special_tokens=False),groups=gs,family=row['family'])
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 docs=sorted(range(2048,2304),key=lambda d:hashlib.sha256(f'H722:{d}'.encode()).hexdigest())[:32];native=[];forwards=0
 for st in range(0,32,8):
  logits,_=run(model,data['all_token_ids'][docs[st:st+8],:64].cuda());native.extend(logits.cpu());forwards+=len(logits)
 records=[];fidelity=[];identityerrors=[];cached_recons={};variants=[('identity',None,None)]+[(kind,seed,ns) for seed in [7241,7242,7243] for kind,ns in [('scalar_top8',None),('initial_raw',None),('learned_raw',None),('initial_random',None),('learned_random',None)]+[(k,n) for k in ['learned_raw_rewired','learned_random_rewired'] for n in [1,2,3]]]
 for f,(layer,head) in sites.items():
  fit=fits[f];mu=fit['mean'].cuda().float();E=fit['write_pca']['encode'].cuda().float();B=fit['write_pca']['decode'].cuda().float()
  keys=[k for k,e in examples.items() if e['family']==f];raw=torch.stack([old['head_coordinates'][k][head] for k in keys]).cuda();vraw=data['activations'][f][docs,7].cuda()
  for kind,seed,ns in variants:
   state=trained[f,'bsf1',seed] if seed is not None else None
   if state is not None:
    sparse=SignedBSF(1).cuda();sparse.load_state_dict(state['model']);sparse.eval()
    M=members[f,seed,kind,ns].cuda() if kind!='scalar_top8' else None
   def reconstruct(h):
    if kind=='identity':return h.clone()
    p=(h-mu)@E
    if kind=='scalar_top8':rp=sparse(p/state['input_scale'])*state['input_scale']
    else:
     z=sparse.encoder(p/state['input_scale']);q=union_pairs(M,select_pairs(z.square(),M));rp=((z*q)@sparse.decoder+sparse.bias)*state['input_scale']
    return mu+rp@B
   rec=reconstruct(raw);vr=reconstruct(vraw);lookup={k:rec[i].cpu() for i,k in enumerate(keys)};cached_recons[f,kind,seed,ns]=dict(task=lookup,validation=vr.cpu());buckets=defaultdict(list)
   for row in rows:
    if row['family']!=f:continue
    for rv,dv in [(0,1),(1,0),(0,2)]:buckets[len(examples[row['id'],rv]['ids'])].append(dict(id=row['id'],family=f,split=row['split'],receiver=rv,donor=dv,kind=kind,seed=seed,null_seed=ns))
   for length,jobs in buckets.items():
    for st in range(0,len(jobs),12):
     bb=jobs[st:st+12];delta=torch.stack([lookup[r['id'],r['donor']]-lookup[r['id'],r['receiver']] for r in bb]).cuda()
     def edit(m,args):
      y=args[0].clone();y[:,-1,head*128:(head+1)*128]+=delta.to(y);return (y,)
     hook=model.blocks[layer].attn.c_proj.register_forward_pre_hook(edit)
     try:logits,_=run(model,torch.tensor([examples[r['id'],r['receiver']]['ids'] for r in bb],device='cuda'))
     finally:hook.remove()
     forwards+=len(bb)
     for n,r in enumerate(bb):
      k=(r['id'],r['receiver']);gs=examples[k]['groups'];base=measure(old['native_logits'][k],gs);ed=measure(logits[n],gs);change=ed['margin']-base['margin'];sign=1 if r['donor'] in [0,2] else -1;records.append(dict(**r,margin_change=change,donor_directed_margin_change=sign*change,edited_donor_correct=ed['top1'][0 if r['donor']==2 else r['donor']]))
      if kind=='identity':prev=next(v for v in oldj['records'] if v['id']==r['id'] and v['receiver']==r['receiver'] and v['donor']==r['donor'] and v['head']==head);identityerrors.append(abs(change-prev['margin_change']))
   for st in range(0,32,8):
    replacement=vr[st:st+8]
    def edit(m,args):
     y=args[0].clone();y[:,-1,head*128:(head+1)*128]=replacement.to(y);return (y,)
    hook=model.blocks[layer].attn.c_proj.register_forward_pre_hook(edit)
    try:logits,_=run(model,data['all_token_ids'][docs[st:st+8],:64].cuda())
    finally:hook.remove()
    forwards+=len(logits)
    for n in range(len(logits)):
     bp=native[st+n].double().log_softmax(-1);ep=logits[n].cpu().double().log_softmax(-1);token=int(data['all_token_ids'][docs[st+n],64]);fidelity.append(dict(site=f,kind=kind,seed=seed,null_seed=ns,doc=docs[st+n],native_to_reconstruction_kl=float((bp.exp()*(bp-ep)).sum()),observed_next_token_ce_change=float(bp[token]-ep[token])))
   print('DONE',f,kind,seed,ns,flush=True);(ART/f'{STEM}.partial.json').write_text(json.dumps(dict(status='running',task_records=records,fidelity=fidelity),indent=2))
 zero=max(r['native_to_reconstruction_kl'] for r in fidelity if r['kind']=='identity');audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['native_to_reconstruction_kl']) for r in fidelity),identity_task_margin_replay_max_abs=max(identityerrors),identity_native_reconstruction_max_kl=zero,whole_model_forwards=forwards,test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(identityerrors)<1e-4 and zero<1e-7;torch.save(dict(reconstructions=cached_recons),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,task_records=records,fidelity=fidelity,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
