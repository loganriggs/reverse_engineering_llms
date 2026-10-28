"""H620 cross-donor complement transfer without fitting new coordinates."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_prefix_cross_donor'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 paths={'source':ART/'bilin18_prefix_diverse_transfer_captures.pt','manifest':ART/'bilin18_prefix_diverse_transfer_manifest.json','basis':ART/'bilin18_prefix_joint_conditional_locked_bases.pt','runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};d=torch.load(paths['source'],weights_only=False);rows=json.loads(paths['manifest'].read_text());B=torch.load(paths['basis'],weights_only=False)['bases']['joint'][:64]
 ex={};seen=set()
 for e in rows:
  if e['wording'] not in [0,2,6,7] or not(e['prefix'] and e['lower']):continue
  if e['prompt'] in seen:continue
  seen.add(e['prompt']);ex[e['id']]=e
 Q={k:x-(x@B.T)@B for k,x in d['components'].items() if k in ex};buckets=defaultdict(list)
 for k,e in ex.items():buckets[e['wording'],len(e['ids'])].append(k)
 union={w:sorted({t for e in ex.values() if e['wording']==w for g in e['groups'] for t in g}) for w in [0,2,6,7]};generic=[tok.encode(s,add_special_tokens=False)[0] for s in [' the',' a',' not']];gen=torch.Generator().manual_seed(620);rand={}
 for k,q in Q.items():
  r=torch.randn(q.shape,generator=gen);r-=r@B.T@B;rand[k]=r*(q.norm()/r.norm())
 jobs=[]
 for keys in buckets.values():
  for k in keys:
   jobs.append((k,k,'native',0.))
   for dose in [.25,1.]:
    jobs.append((k,k,'random',dose))
    for donor in keys:jobs.append((k,donor,'self' if donor==k else 'donor',dose))
 cfg=dict(experiment='H620',hashes=hashes,seed=620,design='Deduplicate recipient prompts before all-donor tests within template and exact token length. H618 contextual interaction complement to frozen H615 joint64. Donor complements scaled to recipient complement total norm. Doses .25/1, self, native and seeded random-complement controls. Copying/agreement template0 anchors; both capital templates. No fit or output filtering.',scope='Total norm matched, not positionwise norm. Donor means entire interaction write from another prompt, not transplant of factual identity. Removal sign fixed. Capital identity evaluated against all8 capitals; random controls one per recipient.',recipient_counts={w:sum(e['wording']==w for e in ex.values()) for w in [0,2,6,7]},evaluations=len(jobs))
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 batches=defaultdict(list)
 for job in jobs:batches[len(ex[job[0]]['ids'])].append(job)
 result=dict(config=cfg,records=[],summary={},status='running');replay=[];normerrs=[]
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 for jj in batches.values():
  for st in range(0,len(jj),16):
   bb=jj[st:st+16];deltas=[]
   for k,donor,mode,dose in bb:
    q=Q[k];x=torch.zeros_like(q) if mode=='native' else rand[k] if mode=='random' else Q[donor]*(q.norm()/Q[donor].norm())
    if mode!='native':normerrs.append(float(abs(x.norm()-q.norm())/q.norm()))
    deltas.append(x*dose)
   delta=torch.stack(deltas)
   def hook(m,args,out):y,b=out;y=y.clone();y[:,4:]-=delta.to(y);return y,b
   h=model.blocks[5].attn.register_forward_hook(hook)
   try:z,_=run(model,torch.tensor([ex[k]['ids'] for k,_,_,_ in bb],device='cuda'))
   finally:h.remove()
   for i,(k,donor,mode,dose) in enumerate(bb):
    zz=z[i].cpu();e=ex[k];de=ex[donor];lp=zz.double().log_softmax(0);nz=d['native'][k];np_=nz.double().log_softmax(0);g=e['groups'][e['variant']];dg=de['groups'][de['variant']];alltokens=union[e['wording']];best=alltokens[int(zz[alltokens].argmax())];margin=torch.logsumexp(zz[g].double(),0)-torch.logsumexp(zz[generic].double(),0);shift=(torch.logsumexp(lp[dg],0)-torch.logsumexp(lp[g],0))-(torch.logsumexp(np_[dg],0)-torch.logsumexp(np_[g],0))
    if mode=='native':replay.append(float((zz-nz).abs().max()))
    result['records'].append(dict(id=k,donor=donor,wording=e['wording'],mode=mode,dose=dose,same_answer=bool(set(g)&set(dg)),answer_token_ids=alltokens,answer_logprobs=lp[alltokens].tolist(),native_answer_logprobs=np_[alltokens].tolist(),recipient_answer_ids=g,donor_answer_ids=dg,kl=float((np_.exp()*(np_-lp)).sum()),top1_correct=int(zz.argmax()) in g,all_answer_choice_correct=best in g,answer_mass=float(lp[g].exp().sum()),conditional_answer_mass=float(torch.exp(torch.logsumexp(lp[g],0)-torch.logsumexp(lp[alltokens],0))),generic_margin=float(margin),donor_relative_margin_delta=float(shift),top_token=tok.decode([int(zz.argmax())]),best_answer_token=tok.decode([best]),component_cosine=float(torch.nn.functional.cosine_similarity(Q[k].flatten(),Q[donor].flatten(),dim=0)) if mode in ['self','donor'] else None))
  save();print('COMPLETE length',len(ex[jj[0][0]]['ids']),flush=True)
 for w in [0,2,6,7]:
  for mode,dose in [('native',0.)]+[(m,t) for t in [.25,1.] for m in ['self','donor','random']]:
   vv=[r for r in result['records'] if r['wording']==w and r['mode']==mode and r['dose']==dose];result['summary'][f'{w}_{mode}_{dose}']=dict(n=len(vv),**{n:float(np.mean([r[n] for r in vv])) for n in ['kl','top1_correct','all_answer_choice_correct','answer_mass','conditional_answer_mass','generic_margin','donor_relative_margin_delta']})
 result['audits']=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},native_replay_max_logit_error=max(replay),norm_relative_error_max=max(normerrs),unique_recipients=len(ex),evaluations=len(result['records']),finite=all(np.isfinite(r[n]) for r in result['records'] for n in ['kl','generic_margin','conditional_answer_mass']));result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
