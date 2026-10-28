"""H954 frozen factorial subject/distractor interventions and native competence."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_relay_contrast_subspaces import run
from run_bilin18_diverse_causal_tasks import groups,measure
from run_bilin18_broad_task_competence import continuation_id
from audit_bilin18_mlp_discriminators import TOKENIZER
from transformers import AutoTokenizer
from pathlib import Path
import torch,json,itertools,numpy as np
import torch.nn.functional as F
S='bilin18_factorial_number_contexts'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'patch':ROOT/'scripts/run_bilin18_relay_contrast_subspaces.py','fits':ART/'bilin18_product_complexity_training_fits.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(v) for k,v in paths.items()}
 cfg={'experiment':'H954','hashes':hashes,'design':'Seven noun pairs, two fixed templates, complete subject number x distractor number grid. All 12 directed swaps per valid quartet, categorized subject-only/distractor-only/both. Fixed preMLP4 near token -3, same absolute position. Relative clause uses invariant past-tense noticed for token alignment. No checkpoint changes or outcome-selected replacements.','competence_gate':'Per template >=4 valid quartets, >=75% quartets correct binary at all four endpoints, >=50% endpoints intended full-vocabulary top1. All valid cases retained.','scope':'Prospective factorial contrasts, not wholly fresh prompts: prepositional opposite-number endpoints intentionally overlap H952. Same source nouns. No fitting; frozen H949 priors.','direction':'recipient/donor encode factorial state indices; historical signed margin field is a replay convention, not necessarily an intended answer change.'}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 nouns=[('pilot','pilots'),('chef','chefs'),('farmer','farmers'),('dancer','dancers'),('singer','singers'),('soldier','soldiers'),('lawyer','lawyers')];manifest=[]
 for fmt in ['prepositional','relative_clause']:
  for k,n in enumerate(nouns):
   texts=[f'The {n[h]} '+(f'with the {d}' if fmt=='prepositional' else f'that the {d} noticed')+' near the window' for h in [0,1] for d in ['guard','guards']]
   manifest.append(dict(id=f'{fmt}-{k}',template=fmt,source_pair=k,texts=texts))
 (ART/f'{S}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');mh=sha(ART/f'{S}_manifest.json');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);ex=[];valid=[];excluded=[]
 for q in manifest:
  ee=[]
  for state,t in enumerate(q['texts']):
   ids=tok.encode(t,add_special_tokens=False);gg=[[continuation_id(tok,t,c) for c in g] for g in groups('agreement',['is','are'])];gg=[[v for v in g if v is not None] for g in gg]
   ee.append(dict(id=q['id'],variant=state,subject_number=state//2,distractor_number=state%2,text=t,ids=ids,groups=gg,position=len(ids)-3))
  if len({len(e['ids']) for e in ee})!=1 or any(tok.decode([e['ids'][e['position']]])!=' near' or any(not g for g in e['groups']) or set(e['groups'][0])&set(e['groups'][1]) for e in ee):excluded.append(q['id']);continue
  q=q|dict(indices=list(range(len(ex),len(ex)+4)));valid.append(q);ex.extend(ee)
 assert valid,'No aligned factorial cells'
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 native={};caps={};identity=0.;actual=0
 for length in sorted({len(e['ids']) for e in ex}):
  inds=[i for i,e in enumerate(ex) if len(e['ids'])==length]
  for st in range(0,len(inds),16):
   ii=inds[st:st+16];ids=torch.tensor([ex[i]['ids'] for i in ii],device='cuda');pos=torch.tensor([ex[i]['position'] for i in ii],device='cuda');x0=F.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None
   for l,block in enumerate(model.blocks):
    raw=block.lambdas[0]*x+block.lambdas[1]*x0;att,bus=block.attn(F.rms_norm(raw,(1152,)),bus);pre=raw+att
    if l==4:cap=pre[torch.arange(len(ii),device='cuda'),pos].cpu()
    x=pre+block.mlp(F.rms_norm(pre,(1152,)))
   y=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1]).float()/30);yp=run(model,ids,cap.cuda(),pos);identity=max(identity,float((y-yp).abs().max()));actual+=2*len(ii)
   for k,i in enumerate(ii):native[i]=y[k].double().cpu();caps[i]=cap[k]
 competence=[]
 for fmt in ['prepositional','relative_clause']:
  qq=[q for q in valid if q['template']==fmt];correct=[];tops=[]
  for q in qq:
   cc=[]
   for i in q['indices']:
    e=ex[i];m=measure(native[i],e['groups']);cc.append(m['margin']>0 if e['subject_number']==0 else m['margin']<0);tops.append(m['top1'][e['subject_number']])
   correct.append(all(cc))
  c=dict(template=fmt,quartets=len(qq),all_four_binary_correct=float(np.mean(correct)) if correct else 0,full_vocab=float(np.mean(tops)) if tops else 0);c['pass']=c['quartets']>=4 and c['all_four_binary_correct']>=.75 and c['full_vocab']>=.5;competence.append(c)
 jobs=[];xs=[];ds=[]
 for q in valid:
  for rec,don in itertools.permutations(range(4),2):
   i,j=q['indices'][rec],q['indices'][don];hc=rec//2!=don//2;dc=rec%2!=don%2;kind='both' if hc and dc else 'subject' if hc else 'distractor'
   jobs.append(dict(id=f"{q['id']}-{rec}-{don}",family='agreement',format=q['template']+':'+kind,template=q['template'],contrast=kind,gap=0,source_pair=q['source_pair'],recipient=rec,donor=don,arm='full',index=i,donor_index=j));xs.append(caps[i]);ds.append(caps[j]-caps[i])
 X=torch.stack(xs);D=torch.stack(ds);mlp=model.blocks[4].mlp
 def truth(z):return torch.cat([mlp(F.rms_norm(b.cuda(),(1152,))).cpu() for b in z.split(32)])
 Y=truth(X);E=truth(X+D)-Y;records=[]
 for length in sorted({len(e['ids']) for e in ex}):
  ns=[n for n,j in enumerate(jobs) if len(ex[j['index']]['ids'])==length]
  for st in range(0,len(ns),16):
   inds=ns[st:st+16];ids=torch.tensor([ex[jobs[n]['index']]['ids'] for n in inds],device='cuda');pos=torch.tensor([ex[jobs[n]['index']]['position'] for n in inds],device='cuda');repl=torch.stack([caps[jobs[n]['donor_index']] for n in inds]).cuda();out=run(model,ids,repl,pos);actual+=len(inds)
   for b,n in enumerate(inds):
    j=jobs[n];i=j['index'];sgn=1 if j['recipient']==1 else -1;records.append(j|dict(kind='full',rank=0,donor_directed_delta=sgn*(measure(out[b],ex[i]['groups'])['margin']-measure(native[i],ex[i]['groups'])['margin'])))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),manifest_unchanged=sha(ART/f'{S}_manifest.json')==mh,identity_logits_max_abs=identity,whole_model_forwards=actual,expected_forwards=2*len(ex)+len(jobs),local_teacher_evaluations=2*len(jobs),finite=bool(torch.isfinite(E).all()),test_evaluations=0,quartets=len(valid),jobs=len(jobs),no_refitting=True)
 assert audits['hashes_unchanged'] and audits['manifest_unchanged'] and identity<1e-4 and audits['finite'] and actual==audits['expected_forwards']
 torch.save(dict(examples=ex,native=native,premlp_relay=caps),ART/f'{S}_captures.pt');torch.save(dict(jobs=jobs,inputs=X,deltas=D,native_updates=Y,true_effects=E),ART/f'{S}_effects.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,competence=competence,excluded=excluded,records=records,audits=audits),indent=2)+'\n');print(competence,excluded,audits,flush=True)
if __name__=='__main__':main()
