"""H621 frozen absolute body-write direction across relations, signs and controls."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_frozen_emission_direction'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);src=ART/'bilin18_prefix_diverse_transfer_captures.pt';bp=ART/'bilin18_prefix_joint_conditional_locked_bases.pt';mp=ART/'bilin18_prefix_diverse_transfer_manifest.json';d=torch.load(src,weights_only=False);B=torch.load(bp,weights_only=False)['bases']['joint'][:64];old=json.loads(mp.read_text());train=[]
 for item in range(4):
  e=next(r for r in old if r['wording']==6 and r['item']==item and r['variant']==0 and r['prefix']==1 and r['lower']==1);train.append(e['id'])
 vv=[d['components'][k]-(d['components'][k]@B.T)@B for k in train];V=torch.stack(vv).mean(0);assert V.shape==(6,1152);gen=torch.Generator().manual_seed(621);random=torch.randn(V.shape,generator=gen);random-=random@B.T@B;random=random/random.norm(dim=-1,keepdim=True)*V.norm(dim=-1,keepdim=True);vectors={'common':V,'random':random,'reversed':V.flip(0)}
 country='France Germany Italy Japan Spain China Russia England'.split();cap='Paris Berlin Rome Tokyo Madrid Beijing Moscow London'.split();lang='French German Italian Japanese Spanish Chinese Russian English'.split()
 panel=[('capital0','the capital city of {} is',country,cap),('capital1','the national capital of {} is',country,cap),('language','the official language of {} is',country,lang),('chemical','the chemical symbol for {} is','oxygen carbon hydrogen nitrogen sulfur phosphorus potassium sodium'.split(),'O C H N S P K Na'.split()),('plural','the plural form of {} is','cat dog book car tree house bird boy'.split(),'cats dogs books cars trees houses birds boys'.split()),('opposite','the opposite meaning of {} is','hot cold big small good bad old young'.split(),'cold hot small big bad good young old'.split())]
 rows=[]
 for family,template,subjects,answers in panel:
  candidates=[]
  for a in answers:
   ids=tok.encode(' '+a,add_special_tokens=False);assert len(ids)==1,(family,a,ids);candidates.append(ids[0])
  for i,(subject,answer) in enumerate(zip(subjects,answers)):
   body=template.format(subject);assert len(tok.encode(' '+body,add_special_tokens=False))==6,(family,body)
   for p in [0,1]:
    for c in [0,1]:
     b=body if c else body[0].upper()+body[1:];prompt=('For the record, ' if p else ' ')+b;rows.append(dict(id=f'{family}-{i}-{p}-{c}',family=family,item=i,subject=subject,answer=answer,answer_id=candidates[i],candidates=candidates,prefix=p,lower=c,ids=tok.encode(prompt,add_special_tokens=False),prompt=prompt,split=('fit' if family=='capital0' and i<4 else 'heldout_identity' if family=='capital0' else 'transfer')))
 manifest=ART/f'{STEM}_manifest.json';manifest.write_text(json.dumps(rows,indent=2));locked=ART/f'{STEM}_locked.pt';torch.save(dict(vectors=vectors,training_ids=train,training_countries=country[:4],direction_design='Mean absolute Q writes; no per-recipient scaling or coefficient estimation'),locked)
 paths={'source':src,'basis':bp,'source_manifest':mp,'manifest':manifest,'vectors':locked,'runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};cfg=dict(experiment='H621',hashes=hashes,training_ids=train,design='Fixed absolute mean complement direction from four countries capital template0, applied to body positions0:6. All six relation scaffolds have six body tokens. Addition/subtraction at .25/1, per-position-norm random complement and reversed-position controls at1. Four native prefix/case conditions retained. No response-based selection or recipient scaling.',scope='Position-dependent six-vector stencil, not one residual vector or proven semantic unit. Previously studied country panel; heldout means excluded from mean fit, not independent history. Relations keep common initial the and final is.',seed=621)
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 native={};result=dict(config=cfg,native=[],interventions=[],summary={},competence={},status='running');generic=tok.encode(' the',add_special_tokens=False)+tok.encode(' a',add_special_tokens=False)+tok.encode(' not',add_special_tokens=False)
 def metrics(z,e):
  lp=z.double().log_softmax(0);ans=e['answer_id'];cs=e['candidates'];best=cs[int(z[cs].argmax())];return dict(top1_correct=int(z.argmax())==ans,candidate_correct=best==ans,answer_mass=float(lp[ans].exp()),conditional_answer_mass=float(torch.exp(lp[ans]-torch.logsumexp(lp[cs],0))),generic_margin=float(z[ans].double()-torch.logsumexp(z[generic].double(),0)),answer_ce=float(-lp[ans]),top_token=tok.decode([int(z.argmax())]))
 for p in [0,1]:
  rr=[e for e in rows if e['prefix']==p]
  for st in range(0,len(rr),16):
   bb=rr[st:st+16];z,_=run(model,torch.tensor([e['ids'] for e in bb],device='cuda'))
   for i,e in enumerate(bb):native[e['id']]=z[i].cpu();result['native'].append(dict(id=e['id'],family=e['family'],prefix=p,lower=e['lower'],**metrics(z[i].cpu(),e)))
 recipients=[e for e in rows if e['prefix'] and e['lower']];modes=[('native',0.)]+[('common',dose) for dose in [-1.,-.25,.25,1.]]+[(kind,dose) for kind in ['random','reversed'] for dose in [-1.,1.]];replay=[]
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 for kind,dose in modes:
  for st in range(0,len(recipients),16):
   bb=recipients[st:st+16];delta=torch.zeros_like(V) if kind=='native' else dose*vectors[kind]
   def hook(m,args,out):y,b=out;y=y.clone();y[:,4:]-=delta.to(y);return y,b
   h=model.blocks[5].attn.register_forward_hook(hook)
   try:z,_=run(model,torch.tensor([e['ids'] for e in bb],device='cuda'))
   finally:h.remove()
   for i,e in enumerate(bb):
    zz=z[i].cpu();nz=native[e['id']];lp=zz.double().log_softmax(0);np_=nz.double().log_softmax(0)
    if kind=='native':replay.append(float((zz-nz).abs().max()))
    result['interventions'].append(dict(id=e['id'],family=e['family'],split=e['split'],kind=kind,dose=dose,kl=float((np_.exp()*(np_-lp)).sum()),**metrics(zz,e)))
  save();print('COMPLETE',kind,dose,flush=True)
 for family,_,_,_ in panel:
  for p in [0,1]:
   for c in [0,1]:
    rr=[r for r in result['native'] if r['family']==family and r['prefix']==p and r['lower']==c];result['competence'][f'{family}_{p}_{c}']={n:float(np.mean([r[n] for r in rr])) for n in ['top1_correct','candidate_correct']}
  for split in (['fit','heldout_identity'] if family=='capital0' else ['transfer']):
   for kind,dose in modes:
    rr=[r for r in result['interventions'] if r['family']==family and r['split']==split and r['kind']==kind and r['dose']==dose];result['summary'][f'{family}_{split}_{kind}_{dose}']={n:float(np.mean([r[n] for r in rr])) for n in ['top1_correct','candidate_correct','answer_mass','conditional_answer_mass','generic_margin','answer_ce','kl']}
 result['audits']=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},native_prompts=len(rows),interventions=len(result['interventions']),native_replay_max_logit_error=max(replay),training_ids=train,finite=all(np.isfinite(r[n]) for r in result['interventions'] for n in ['kl','answer_ce','generic_margin']));result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
