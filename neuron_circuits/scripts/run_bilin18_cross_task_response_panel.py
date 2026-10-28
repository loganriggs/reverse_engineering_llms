"""H633: frozen cross-task finite-response atlas and context-transfer baselines."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_cross_task_response_panel'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 def token(s):
  ids=tok.encode(' '+s,add_special_tokens=False);assert len(ids)==1,(s,ids);return ids[0]
 words='cat dog book car tree house bird boy'.split();countries='France Germany Italy Japan Spain China Russia England'.split();caps='Paris Berlin Rome Tokyo Madrid Beijing Moscow London'.split()
 definitions={
 'copy':(words,words,['Repeat the word: {}. The word is','Copy exactly: {}\nCopy:'],words),
 'fact':(countries,caps,['The capital of {} is','What is the capital of {}?\nAnswer:'],caps),
 'plural':(words,[w+'s' for w in words],['The plural of {} is','Singular: {}\nPlural:'],words+[w+'s' for w in words]),
 'agreement':(['The cat','The dogs','The boy','The girls','The bird','The cars','The tree','The books'],['is','are']*4,['{}','Complete with is or are: {}'],['is','are']),
 'addition':(['1 + 1','2 + 1','2 + 2','3 + 2','3 + 3','4 + 3','4 + 4','5 + 4'],[str(n) for n in range(2,10)],['{} =','Calculate: {}\nAnswer:'],[str(n) for n in range(10)])}
 rows=[]
 for family,(subjects,answers,templates,candidates) in definitions.items():
  for form,template in enumerate(templates):
   for item,(subject,answer) in enumerate(zip(subjects,answers)):
    prompt=template.format(subject);rows.append(dict(id=f'{family}-{form}-{item}',family=family,form=form,item=item,prompt=prompt,ids=tok.encode(prompt,add_special_tokens=False),answer=answer,answer_id=token(answer),candidate_ids=[token(c) for c in candidates],split='fit' if form==0 and item<4 else 'heldout'))
 mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2));pp=ART/'bilin18_emission_probe_directions_locked.pt';probes=torch.load(pp,weights_only=False)
 paths={'manifest':mp,'probes':pp,'runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()}
 cfg=dict(experiment='H633',hashes=hashes,design='Five task families x two constructions x eight items, nine frozen H632 probes. Fixed first four items of template0 fit task/global mean full-logit derivatives. All remaining contexts held out. No prompt filtering. Central +/- .02 responses; original/span0/orthogonal0 checked at half step. Original finite +/-.1 KL calibration.',limitations='Context-transfer baseline only; does not test unseen directions or fit low rank. First-token exact scoring includes explicit competitors; generic agreement continuation is not forced binary completion.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 buckets=defaultdict(list)
 for i,e in enumerate(rows):buckets[len(e['ids'])].append(i)
 forwards=0
 def evaluate(q=None,alpha=0.):
  nonlocal forwards
  out=torch.empty(len(rows),model.lm_head.weight.shape[0])
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];h=None
    if q is not None:
     def edit(m,args,value):y,b=value;y=y.clone();y[:,-1]+=alpha*q.to(y);return y,b
     h=model.blocks[5].attn.register_forward_hook(edit)
    try:z,_=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'))
    finally:
     if h is not None:h.remove()
    out[bb]=z.cpu();forwards+=len(bb)
  return out
 native=evaluate();responses={};validation=[];finite={}
 for name,q in probes.items():
  minus=evaluate(q,-.02);plus=evaluate(q,.02);responses[name]=(plus.double()-minus.double())/.04
  if name in ['original','span0','orthogonal0']:
   half=(evaluate(q,.01).double()-evaluate(q,-.01).double())/.02
   for i,e in enumerate(rows):
    p=native[i].double().softmax(0)
    def norm(z):return ((z-(p*z).sum()).square()*p).sum().sqrt()
    validation.append(dict(id=e['id'],probe=name,relative_error=float(norm(responses[name][i]-half[i])/norm(half[i]))))
  if name=='original':
   for alpha in [-.1,.1]:finite[alpha]=evaluate(q,alpha)
  print('COMPLETE',name,'forwards',forwards,flush=True)
 competence=[];records=[];kl=[]
 fit=[i for i,e in enumerate(rows) if e['split']=='fit'];taskfit={f:[i for i in fit if rows[i]['family']==f] for f in definitions}
 means={name:{'global':G[fit].mean(0),**{f:G[ii].mean(0) for f,ii in taskfit.items()}} for name,G in responses.items()}
 for i,e in enumerate(rows):
  z=native[i].double();p=z.softmax(0);cs=e['candidate_ids'];competence.append(dict(id=e['id'],family=e['family'],form=e['form'],split=e['split'],top1_correct=int(z.argmax())==e['answer_id'],candidate_correct=cs[int(z[cs].argmax())]==e['answer_id'],target_probability=float(p[e['answer_id']]),top_token=tok.decode([int(z.argmax())])))
  def norm(x):return ((x-(p*x).sum()).square()*p).sum().sqrt()
  for name,G in responses.items():
   g=G[i]
   for method in ['global','task']:
    pred=means[name]['global' if method=='global' else e['family']];records.append(dict(id=e['id'],family=e['family'],form=e['form'],split=e['split'],probe=name,method=method,native_norm=float(norm(g)),relative_error=float(norm(pred-g)/norm(g)),target_derivative=float(g[e['answer_id']]-(p*g).sum())))
  for alpha,actual in finite.items():
   truth=actual[i].double().log_softmax(0);pt=truth.exp()
   for method,g in [('own',responses['original'][i]),('task',means['original'][e['family']]),('global',means['original']['global']),('zero',torch.zeros_like(z))]:
    pred=(z+alpha*g).log_softmax(0);kl.append(dict(id=e['id'],family=e['family'],form=e['form'],split=e['split'],alpha=alpha,method=method,kl=float((pt*(truth-pred)).sum())))
 summary={}
 for f in definitions:
  for form in range(2):
   cc=[r for r in competence if r['family']==f and r['form']==form and r['split']=='heldout'];s={'n':len(cc),'top1':np.mean([r['top1_correct'] for r in cc]),'candidate':np.mean([r['candidate_correct'] for r in cc])}
   for method in ['global','task']:
    rr=[r for r in records if r['family']==f and r['form']==form and r['split']=='heldout' and r['method']==method];s[method+'_relative_error']=np.mean([r['relative_error'] for r in rr])
   for method in ['own','task','global','zero']:
    rr=[r for r in kl if r['family']==f and r['form']==form and r['split']=='heldout' and r['method']==method];s[method+'_kl']=np.mean([r['kl'] for r in rr])
   summary[f'{f}_{form}']=s
 audits=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},native_forwards=forwards,half_step_max_error=max(r['relative_error'] for r in validation),finite=all(np.isfinite(r['relative_error']) for r in records),fit_n=len(fit),heldout_n=len(rows)-len(fit))
 torch.save(dict(native_logits=native,responses=responses,finite_logits=finite,means=means),ART/f'{STEM}_captures.pt');result=dict(status='complete',config=cfg,competence=competence,records=records,validation=validation,kl=kl,summary=summary,audits=audits,elapsed_seconds=time.time()-start);(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print('AUDITS',audits,flush=True);print('SUMMARY',summary,flush=True)
if __name__=='__main__':main()
