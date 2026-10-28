"""H966: fixed factorial valence/negation panel with frozen H965 directions."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import groups,measure,forward
from run_bilin18_broad_task_competence import continuation_id
from audit_bilin18_mlp_discriminators import TOKENIZER
from transformers import AutoTokenizer
from pathlib import Path
from collections import defaultdict
import torch,json,time,numpy as np
S='bilin18_factorial_valence_negation'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'runner':Path(__file__),'directions':ART/'bilin18_shared_task_directions_directions.pt','model':SNAPSHOT/'pytorch_model.bin','forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','oldscreen':ART/'bilin18_expanded_task_bank.json'};hashes={k:sha(p) for k,p in paths.items()};old=json.loads(paths['oldscreen'].read_text());oldtexts={t for r in old['records'] for t in r['prompts']};directions=torch.load(paths['directions'],weights_only=False)['directions'];adjs=[('wonderful','terrible'),('excellent','awful'),('great','bad'),('pleasant','unpleasant'),('superb','horrible'),('fantastic','abysmal'),('delightful','dreadful'),('impressive','atrocious')]
 cfg={'experiment':'H966','hashes':hashes,'design':'8 adjective pairs x3 output contexts x4 states (positive/negative adjective crossed with absent/present not). Frozen H965 sentiment/negation directions; no fitting. All12 directed edges per quartet, with full-write, identity, both frozen rank1 projections and their complements at final-token MLP12.','native_gate':'Per format and adjective cohort: >=4 quartets, >=.75 all-four binary correct, >=.5 pooled full-vocabulary intended group top1. No individual native-correct filtering. Failed cohorts remain diagnostics.','factors':'a=+1 positive adjective,-1 negative; n=+1 plain,-1 not. Ideal binary valence=a*n is an operational expectation, not a claim that linguistic negation always reverses evaluative polarity. In particular not bad can be weakly positive or neutral.','cohorts':'First4 adjective pairs appear in H965 discovery; last4 adjective pairs absent from H965 task-bank examples. All directions frozen. Some familiar format A prompts deliberately overlap prior data; explicit overlap counts. Not the original reserved test.','interpretation':'Hadamard decomposition of four writes separates lexical adjective (a), presence of not (n), and interaction (a*n). Interaction is a necessary signature of ideal valence reversal on this panel but is not sufficient evidence of a reusable negation operator.','fidelity':'KL(full donor write patch||candidate), relative to native baseline; projection coefficient uses true donor delta. Counterfactual span adequacy only; no deployable encoder or unsupervised-manifold claim.'}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);ex=[];quartets=[];excluded=[]
 for fmt in ['A','B','C']:
  for source,(pos,neg) in enumerate(adjs):
   q=dict(id=f'{fmt}-{source}',format=fmt,source=source,cohort='familiar_adjectives' if source<4 else 'new_to_H965_adjectives',indices=[])
   for a,n in [(1,1),(1,-1),(-1,1),(-1,-1)]:
    phrase=('not ' if n==-1 else '')+(pos if a==1 else neg)
    text=f'Review: The film was {phrase}. Sentiment:' if fmt=='A' else (f'Review: The meal was {phrase}. Overall sentiment:' if fmt=='B' else f'The reviewer described the performance as {phrase}. The review is')
    gg=[[continuation_id(tok,text,s) for s in g] for g in groups('sentiment',['positive','negative'])];gg=[[v for v in g if v is not None] for g in gg]
    if any(not g for g in gg):excluded.append(dict(id=q['id'],a=a,n=n,reason='answer boundary'));continue
    q['indices'].append(len(ex));ex.append(dict(text=text,ids=tok.encode(text,add_special_tokens=False),groups=gg,a=a,n=n,valence=a*n,prior_prompt=text in oldtexts))
   if len(q['indices'])==4:quartets.append(q)
 assert not excluded and len(ex)==96 and len(quartets)==24 and len({e['text'] for e in ex})==96
 (ART/f'{S}_manifest.json').write_text(json.dumps(dict(config=cfg,examples=ex,quartets=quartets),indent=2)+'\n');mh=sha(ART/f'{S}_manifest.json')
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 native={};writes={};actual=0
 for length in sorted({len(e['ids']) for e in ex}):
  ii=[i for i,e in enumerate(ex) if len(e['ids'])==length]
  for st in range(0,len(ii),24):
   inds=ii[st:st+24];y,cap=forward(model,torch.tensor([ex[i]['ids'] for i in inds],device='cuda'),capture=True);actual+=len(inds)
   for k,i in enumerate(inds):native[i]=y[k].double().cpu();writes[i]=cap[12][k]
 native_records=[];factorial=[]
 H=torch.tensor([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],dtype=torch.float64)
 factor_replay=0.
 for q in quartets:
  mm=[measure(native[i],ex[i]['groups']) for i in q['indices']];correct=[m['margin']*ex[i]['valence']>0 for m,i in zip(mm,q['indices'])];top=[m['top1'][0 if ex[i]['valence']==1 else 1] for m,i in zip(mm,q['indices'])]
  native_records.append(q|dict(all_four_correct=all(correct),binary_correct=correct,full_vocab=sum(top)/4,margins=[m['margin'] for m in mm]))
  X=torch.stack([writes[i] for i in q['indices']]).double();C=H.T@X/4;replay=X.mean(0)+H@C;factor_replay=max(factor_replay,float((X-replay).abs().max()));energy=C.square().sum(1);M=torch.tensor([m['margin'] for m in mm],dtype=torch.float64);factorial.append(q|dict(write_factor_energy_fraction=(energy/energy.sum()).tolist(),native_margin_coefficients=(H.T@M/4).tolist(),direction_coefficients={name:(C@directions[name].double()).tolist() for name in ['sentiment','negation']}))
 competence=[]
 for fmt in ['A','B','C']:
  for cohort in ['familiar_adjectives','new_to_H965_adjectives']:
   rr=[r for r in native_records if r['format']==fmt and r['cohort']==cohort];both=float(np.mean([r['all_four_correct'] for r in rr]));fv=float(np.mean([r['full_vocab'] for r in rr]));competence.append(dict(format=fmt,cohort=cohort,n=len(rr),all_four_correct=both,full_vocab=fv,pass_gate=len(rr)>=4 and both>=.75 and fv>=.5))
 full={};records=[];ident=0.
 def execute(jobs,isfull=False):
  nonlocal actual,ident
  buckets=defaultdict(list)
  for j in jobs:buckets[len(ex[j['i']]['ids'])].append(j)
  for length,jj in sorted(buckets.items()):
   for st in range(0,len(jj),24):
    batch=jj[st:st+24];y,_=forward(model,torch.tensor([ex[j['i']]['ids'] for j in batch],device='cuda'),patch=(12,torch.stack([writes[j['i']]+j['edit'] for j in batch])));actual+=len(batch)
    for k,j in enumerate(batch):
     out=y[k].double().cpu();key=j['q']['id'],j['rec'],j['don'];base=native[j['i']]
     if isfull:full[key]=out
     ref=full[key];lr=ref.log_softmax(0);lp=out.log_softmax(0);ln=base.log_softmax(0);bm=measure(base,ex[j['i']]['groups'])['margin'];om=measure(out,ex[j['i']]['groups'])['margin'];fm=measure(ref,ex[j['i']]['groups'])['margin']
     if j['arm']=='identity':ident=max(ident,float((out-base).abs().max()))
     records.append({k:j['q'][k] for k in ['id','format','source','cohort']}|dict(recipient=j['rec'],donor=j['don'],edge=j['edge'],recipient_valence=ex[j['i']]['valence'],donor_valence=ex[j['di']]['valence'],arm=j['arm'],direction=j['direction'],margin_change=om-bm,full_margin_change=fm-bm,KL=float((lr.exp()*(lr-lp)).sum()),baseline_KL=float((lr.exp()*(lr-ln)).sum()),edit_norm=float(j['edit'].norm())))
 refs=[];jobs=[]
 for q in quartets:
  for rec,i in enumerate(q['indices']):
   for don,di in enumerate(q['indices']):
    if rec==don:continue
    assert ex[i]['ids'][-1]==ex[di]['ids'][-1]
    ac=ex[i]['a']!=ex[di]['a'];nc=ex[i]['n']!=ex[di]['n'];edge='both' if ac and nc else ('adjective' if ac else 'negation');delta=writes[di]-writes[i];b=dict(q=q,rec=rec,don=don,i=i,di=di,edge=edge);refs.append(b|dict(arm='full',direction='full',edit=delta));jobs.append(b|dict(arm='identity',direction='identity',edit=delta*0))
    for name in ['sentiment','negation']:
     u=directions[name];p=(delta@u)*u
     jobs.extend([b|dict(arm='projection',direction=name,edit=p),b|dict(arm='complement',direction=name,edit=delta-p)])
 execute(refs,True);execute(jobs)
 summary=[];grouped=defaultdict(list)
 for r in records:grouped[r['format'],r['cohort'],r['edge'],r['arm'],r['direction']].append(r)
 for key,rr in grouped.items():
  b=sum(r['baseline_KL'] for r in rr);summary.append(dict(zip(['format','cohort','edge','arm','direction'],key))|dict(n=len(rr),KL_ratio=sum(r['KL'] for r in rr)/b,margin_change_rms=float(np.mean([r['margin_change']**2 for r in rr])**.5),full_margin_change_rms=float(np.mean([r['full_margin_change']**2 for r in rr])**.5),margin_error_rms=float(np.mean([(r['margin_change']-r['full_margin_change'])**2 for r in rr])**.5)))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),manifest_unchanged=sha(ART/f'{S}_manifest.json')==mh,whole_model_forwards=actual,expected_forwards=len(ex)+len(refs)+len(jobs),identity_logits_max_abs=ident,factorial_reconstruction_max_abs=factor_replay,finite=all(np.isfinite(r['KL']) for r in records),prior_prompt_overlap=sum(e['prior_prompt'] for e in ex),test_evaluations=0,no_training=True)
 assert audits['hashes_unchanged'] and audits['manifest_unchanged'] and actual==audits['expected_forwards'] and ident<1e-4 and factor_replay<1e-10 and audits['finite']
 torch.save(dict(examples=ex,quartets=quartets,native=native,writes=writes),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,competence=competence,native_records=native_records,factorial=factorial,records=records,summary=summary,excluded=excluded,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(competence,audits,flush=True)
if __name__=='__main__':main()
