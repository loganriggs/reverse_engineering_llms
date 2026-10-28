"""H1002: frozen noun direction, crossed background cue and action probes."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward,measure
from run_bilin18_broad_task_competence import continuation_id
from audit_bilin18_mlp_discriminators import TOKENIZER
from transformers import AutoTokenizer
from pathlib import Path
from collections import defaultdict
import torch,json,time,numpy as np
S='bilin18_noun_cue_controls'
def stimuli():
 pairs=[];gg=[[' bread',' meat',' cake',' cheese',' vegetables',' fruit',' apple'],[' door',' gate',' car',' lock',' house',' box']]
 for person in ['pilot','chef','farmer','singer']:
  for cohort,verbs in [('familiar',('cut','unlocked')),('reworded',('sliced','opened')),('novel_eat_enter',('ate','entered')),('novel_cook_lock',('cooked','locked'))]:
   cells=[[f'A {tool} lay nearby. The {person} {verb} the' for verb in verbs] for tool in ['knife','key']]
   for ti in range(2):pairs.append(dict(id=f'{person}-{cohort}-action-{ti}',cohort=cohort,kind='action',condition=['knife','key'][ti],prompts=cells[ti],intended=[0,1],group_strings=gg))
   for ai in range(2):pairs.append(dict(id=f'{person}-{cohort}-nuisance-{ai}',cohort=cohort,kind='background_tool',condition=str(ai),prompts=[cells[ti][ai] for ti in range(2)],intended=[ai,ai],group_strings=gg))
  for cohort,ps in [('novel_tools_eat',[f'Using a fork, the {person} ate the',f'Using a card, the {person} entered the']),('novel_tools_taste',[f'Using a spoon, the {person} tasted the',f'Using a code, the {person} accessed the'])]:pairs.append(dict(id=f'{person}-{cohort}',cohort=cohort,kind='action',condition='coupled_new_tools',prompts=ps,intended=[0,1],group_strings=gg))
 return pairs
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','bases':ART/'bilin18_residual_discovery_spans_bases.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};bd=torch.load(paths['bases'],weights_only=False);B=bd['pca'];q=bd['bases']['noun_expectation_rank1'];rows=stimuli()
 cfg=dict(experiment='H1002',hashes=hashes,design='Frozen H1001nounrank1. Cross background knife/key with food/access actions, four subjects, four verb pairs. Action swaps hold background fixed; background swaps hold action fixed. Additional new tool/action pairs are explicitly coupled. No fitting or outcome-selected prompts.',gate='For each action cohort/condition:4pairs, >=.75both intended endpoints category-correct and >=.5full-vocab top1. No correctness filtering. Nuisance pairs have same intended category at both endpoints.',limits='Background nouns may legitimately prime continuations. Category lists imperfect; food versus access is not a universal semantic taxonomy. New-tool pairs covary tool and action. Original subjects reused, prompts new. No claim of complete factorial orthogonalization of meaning.',controls='Identity, full donor, nounrank1, task128, task128+nounrank1, paired +/- residual noise norm matched to nounrank1 edit. No original holdouts.',training=False)
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');(ART/f'{S}_manifest.json').write_text(json.dumps(rows,indent=2)+'\n');mh=sha(ART/f'{S}_manifest.json');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);ex=[];lookup={}
 for r in rows:
  r['indices']=[]
  for text in r['prompts']:
   if text not in lookup:
    gs=[[continuation_id(tok,text,s) for s in group] for group in r['group_strings']];assert all(all(t is not None for t in g) for g in gs);lookup[text]=len(ex);ex.append(dict(text=text,ids=tok.encode(text,add_special_tokens=False),groups=gs))
   r['indices'].append(lookup[text])
  assert ex[r['indices'][0]]['ids'][-1]==ex[r['indices'][1]]['ids'][-1]
 assert len(ex)==80 and len(rows)==72
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 native={};writes={};actual=0
 for length in sorted({len(e['ids']) for e in ex}):
  ii=[i for i,e in enumerate(ex) if len(e['ids'])==length]
  for st in range(0,len(ii),24):
   inds=ii[st:st+24];y,c=forward(model,torch.tensor([ex[i]['ids'] for i in inds],device='cuda'),capture=True);actual+=len(inds)
   for k,i in enumerate(inds):native[i]=y[k].double().cpu();writes[i]=c[12][k].double()
 competence=[]
 for r in rows:
  ms=[measure(native[i],ex[i]['groups']) for i in r['indices']];r['both_correct']=all((m['margin']>0)==(target==0) for m,target in zip(ms,r['intended']));r['full_vocab']=sum(m['top1'][target] for m,target in zip(ms,r['intended']))/2
 for cohort,condition in sorted({(r['cohort'],r['condition']) for r in rows if r['kind']=='action'}):
  rr=[r for r in rows if r['kind']=='action' and (r['cohort'],r['condition'])==(cohort,condition)];both=np.mean([r['both_correct'] for r in rr]);full=np.mean([r['full_vocab'] for r in rr]);competence.append(dict(cohort=cohort,condition=condition,n=len(rr),both_correct=float(both),full_vocab=float(full),pass_gate=bool(len(rr)>=4 and both>=.75 and full>=.5)))
 jobs=defaultdict(list)
 for r in rows:
  for rec,i in enumerate(r['indices']):
   donor=r['indices'][1-rec];delta=writes[donor]-writes[i];task=B@(B.T@delta);one=q@(q.T@delta);noise=torch.randn(1152,dtype=torch.double,generator=torch.Generator().manual_seed(100200+i));noise-=B@(B.T@noise);noise=noise/noise.norm()*one.norm()
   edits={'identity':torch.zeros_like(delta),'donor':delta,'noun1':one,'task128':task,'augmented129':task+one,'random_plus':noise,'random_minus':-noise}
   for arm,edit in edits.items():jobs[len(ex[i]['ids'])].append((r,rec,i,donor,arm,(writes[i]+edit).float(),float(edit.norm()),float((q.T@delta)[0])))
 records=[];logits=[];identity=0.
 for length,jj in sorted(jobs.items()):
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];y,_=forward(model,torch.tensor([ex[t[2]]['ids'] for t in batch],device='cuda'),patch=(12,torch.stack([t[5] for t in batch])));actual+=len(batch)
   for k,(r,rec,i,donor,arm,repl,norm,coord) in enumerate(batch):
    out=y[k].double().cpu();m=measure(out,ex[i]['groups'])['margin'];bm=measure(native[i],ex[i]['groups'])['margin'];lp=out.log_softmax(0);ln=native[i].log_softmax(0)
    if arm=='identity':identity=max(identity,float((out-native[i]).abs().max()))
    records.append(dict(id=r['id'],cohort=r['cohort'],condition=r['condition'],pair_kind=r['kind'],recipient=rec,index=i,donor=donor,kind=arm,margin_change=m-bm,effect=(1 if rec==1 else -1)*(m-bm),edit_norm=norm,coordinate_delta=coord,KL_to_native=float((lp.exp()*(lp-ln)).sum())));logits.append(out.float())
  print('DONE',length,actual,flush=True)
 refs={(r['id'],r['recipient']):logits[k].double().log_softmax(0) for k,r in enumerate(records) if r['kind']=='donor'}
 for k,r in enumerate(records):
  ref=refs[r['id'],r['recipient']];lp=logits[k].double().log_softmax(0);r['full_patch_KL']=float((ref.exp()*(ref-lp)).sum())
 grouped=defaultdict(list)
 for r in records:grouped[r['cohort'],r['pair_kind'],r['kind']].append(r)
 summary=[dict(zip(['cohort','pair_kind','kind'],key))|dict(n=len(rr),mean_effect=float(np.mean([r['effect'] for r in rr])),mean_absolute_margin_change=float(np.mean([abs(r['margin_change']) for r in rr])),mean_full_patch_KL=float(np.mean([r['full_patch_KL'] for r in rr])),mean_edit_norm=float(np.mean([r['edit_norm'] for r in rr]))) for key,rr in grouped.items()]
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),manifest_unchanged=sha(ART/f'{S}_manifest.json')==mh,whole_model_forwards=actual,expected_forwards=1088,identity_logits_max_abs=identity,no_training=True,test_evaluations=0,finite=all(np.isfinite(r['effect']) and np.isfinite(r['full_patch_KL']) for r in records))
 assert audits['hashes_unchanged'] and audits['manifest_unchanged'] and actual==1088 and identity<1e-4 and audits['finite']
 torch.save(dict(examples=ex,native=native,writes=writes,logits=torch.stack(logits)),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,pairs=rows,competence=competence,records=records,summary=summary,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(competence,audits,flush=True)
if __name__=='__main__':main()
