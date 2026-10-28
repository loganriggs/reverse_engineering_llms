"""H623 fixed final vector across surfaces; disjoint answer-prefix scoring."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_emission_surface_transfer'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);src=ART/'bilin18_frozen_emission_direction_locked.pt';bp=ART/'bilin18_prefix_joint_conditional_locked_bases.pt';v=torch.load(src,weights_only=False)['vectors']['common'][-1];B=torch.load(bp,weights_only=False)['bases']['joint'][:64];g=torch.Generator().manual_seed(623);random=torch.randn(v.shape,generator=g);random-=random@B.T@B;random*=v.norm()/random.norm()
 countries='France Germany Italy Japan Spain China Russia England'.split();data=[('capital',countries,'Paris Berlin Rome Tokyo Madrid Beijing Moscow London'.split(),['For the record, the capital city of {} is',"{}'s capital is",'What is the capital of {}?\nAnswer:']),('language',countries,'French German Italian Japanese Spanish Chinese Russian English'.split(),['For the record, the official language of {} is','People in {} speak','What language is spoken in {}?\nAnswer:']),('plural','cat dog book car tree house bird boy'.split(),'cats dogs books cars trees houses birds boys'.split(),['For the record, the plural form of {} is',"The plural of '{}' is",'What is the plural of {}?\nAnswer:'])];rows=[];queries={}
 for family,subjects,answers,forms in data:
  for form,template in enumerate(forms):
   for i,subject in enumerate(subjects):
    prompt=template.format(subject);ids=tok.encode(prompt,add_special_tokens=False);variants=[]
    for ans in answers:
     vv=[]
     for text in [' '+ans,' "'+ans]:
      full=tok.encode(prompt+text,add_special_tokens=False);assert full[:len(ids)]==ids;vv.append(full[len(ids):])
     variants.append(vv)
    seqs=[x for vv in variants for x in vv];assert len({tuple(x) for x in seqs})==len(seqs)
    for a in seqs:
     for b in seqs:
      if a!=b:assert not (len(a)<=len(b) and b[:len(a)]==a)
    e=dict(id=f'{family}-{form}-{i}',family=family,form=form,item=i,subject=subject,answers=answers,prompt=prompt,ids=ids,variants=variants);idx=len(rows);rows.append(e)
    for seq in seqs:
     for j,t in enumerate(seq):queries.setdefault((idx,tuple(seq[:j])),set()).add(t)
 mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2));vp=ART/f'{STEM}_locked.pt';torch.save(dict(common=v,random=random),vp);paths={'source':src,'basis':bp,'manifest':mp,'vectors':vp,'runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};modes=[('native',0.),('common',-1.),('common',.25),('common',1.),('random',-1.),('random',1.)];cfg=dict(experiment='H623',hashes=hashes,seed=623,design='Freeze H621 final-position absolute vector. Three relations by three surface forms by eight items. Patch original prompt final position only, including teacher-forced continuation forwards. Score direct and opening-double-quote answer prefixes, sum disjoint event probabilities per identity. Six signed modes; no fitting, scaling or competence filtering.',scope='Finite answer-prefix events, not complete semantic correctness or free generation. Quote prefix stops after answer and does not require closing quote; no arbitrary explanations included. Random one seeded complement direction. Original country panel adaptive.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 buckets=defaultdict(list)
 for key in queries:buckets[len(rows[key[0]]['ids'])+len(key[1])].append(key)
 native={};result=dict(config=cfg,records=[],summary={},status='running');generic=[tok.encode(s,add_special_tokens=False)[0] for s in [' the',' a',' not']];total=0
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 for kind,dose in modes:
  edges={};roots={};delta=torch.zeros_like(v) if kind=='native' else dose*(v if kind=='common' else random)
  for keys in buckets.values():
   for st in range(0,len(keys),32):
    bb=keys[st:st+32];positions=torch.tensor([len(rows[i]['ids'])-1 for i,p in bb],device='cuda')
    def hook(m,args,out):y,b=out;y=y.clone();y[torch.arange(len(bb),device=y.device),positions]-=delta.to(y);return y,b
    h=model.blocks[5].attn.register_forward_hook(hook)
    try:z,_=run(model,torch.tensor([rows[i]['ids']+list(p) for i,p in bb],device='cuda'))
    finally:h.remove()
    total+=len(bb)
    for n,key in enumerate(bb):
     zz=z[n].cpu();lp=zz.double().log_softmax(0);edges[key]={t:float(lp[t]) for t in queries[key]}
     if not key[1]:roots[key[0]]=zz
  for i,e in enumerate(rows):
   scores=[];direct=[];quoted=[]
   for vv in e['variants']:
    ss=[sum(edges[i,tuple(seq[:j])][t] for j,t in enumerate(seq)) for seq in vv];direct.append(ss[0]);quoted.append(ss[1]);scores.append(float(torch.logsumexp(torch.tensor(ss,dtype=torch.float64),0)))
   z=roots[i];lp=z.double().log_softmax(0)
   if kind=='native':native[i]=z
   np_=native[i].double().log_softmax(0);sc=torch.tensor(scores,dtype=torch.float64);ans=e['item'];directids=e['variants'][ans][0];result['records'].append(dict(id=e['id'],family=e['family'],form=e['form'],item=ans,kind=kind,dose=dose,prompt_tokens=len(e['ids']),last_token=tok.decode([e['ids'][-1]]),kl=float((np_.exp()*(np_-lp)).sum()),direct_top1=(int(z.argmax())==directids[0] if len(directids)==1 else None),direct_choice=int(np.argmax(direct))==ans,combined_choice=int(sc.argmax())==ans,direct_mass=float(np.exp(direct[ans])),quoted_mass=float(np.exp(quoted[ans])),combined_mass=float(sc[ans].exp()),conditional_mass=float(sc.softmax(0)[ans]),generic_mass=float(lp[generic].exp().sum()),top_token=tok.decode([int(z.argmax())]),candidate_logprobs=scores))
  save();print('COMPLETE',kind,dose,flush=True)
 for family,_,_,_ in data:
  for form in range(3):
   for kind,dose in modes:
    rr=[r for r in result['records'] if r['family']==family and r['form']==form and r['kind']==kind and r['dose']==dose];result['summary'][f'{family}_{form}_{kind}_{dose}']={n:float(np.mean([r[n] for r in rr])) for n in ['kl','direct_top1','direct_choice','combined_choice','direct_mass','quoted_mass','combined_mass','conditional_mass','generic_mass']}
 result['audits']=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},prompts=len(rows),condition_records=len(result['records']),unique_teacher_prefix_queries=len(queries),model_sequence_evaluations=total,finite=all(np.isfinite(r[n]) for r in result['records'] for n in ['kl','combined_mass','conditional_mass']),random_norm_relative_error=float(abs(random.norm()-v.norm())/v.norm()));result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
