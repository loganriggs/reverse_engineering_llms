"""H980 fresh competence panel for frozen single-state model."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import groups,measure,forward
from run_bilin18_broad_task_competence import continuation_id
from audit_bilin18_mlp_discriminators import TOKENIZER
from transformers import AutoTokenizer
from pathlib import Path
import torch,json,numpy as np,time
S='bilin18_fresh_single_state_panel'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'prior':ART/'bilin18_factorial_valence_negation_captures.pt','bank':ART/'bilin18_expanded_task_bank.json','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};prior=torch.load(paths['prior'],weights_only=False);prior_text={e['text'] for e in prior['examples']};bank=json.loads(paths['bank'].read_text());prior_text.update(t for r in bank['records'] for t in r['prompts']);known_adjectives={word for r in bank['records'] if r['family']=='sentiment' for t in r['prompts'] for word in [t.split(' was ')[1].split('.')[0]]}
 adjs=[('absorbing','tiresome'),('captivating','lifeless'),('inspiring','depressing'),('imaginative','formulaic'),('refreshing','stale'),('polished','sloppy'),('stunning','hideous'),('charming','annoying'),('uplifting','miserable'),('exquisite','repulsive'),('moving','forgettable'),('enchanting','obnoxious')]
 earlier=json.loads((ART/'bilin18_prospective_composition_panel_manifest.json').read_text());prior_text.update(e['text'] for e in earlier['examples']);known_adjectives.update(a for q in earlier['quartets'] for a in q['adjectives']);assert not any(a in known_adjectives for pair in adjs for a in pair)

 cfg={'experiment':'H980','hashes':hashes,'design':'12 new adjective pairs in three fixed cohorts x4 templates x4 states. Templates film,book,story and changed I-thought review prefix. No outcome-based replacement; model H979 already frozen.','gate':'At least4 quartets/cohort; >=.75 all-four binary correctness and >=.5 full-vocabulary intended-group top1.','frozen_model_sha256':sha(ART/'bilin18_single_state_composition_models.pt'),'novelty':'Adjective strings checked against H962/H975 panels; exact prompts checked against H962/H966/H975. No claim of unseen model-training words.','capture':'Native logits and normalized adjective MLP3 inputs/writes; no fitting, interventions or reserved-test access.','semantics':'Binary negation reversal is an operational test; not-negative may be neutral.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);ex=[];qs=[];excluded=[]
 for fmt in ['film','book','story','thought']:
  for source,(pos,neg) in enumerate(adjs):
   q=dict(id=f'{fmt}-{source}',format=fmt,source=source,cohort=f'fresh{source//4+1}',adjectives=[pos,neg],indices=[])
   for a,n in [(1,1),(1,-1),(-1,1),(-1,-1)]:
    noun='film' if fmt in ['film','thought'] else fmt;phrase=('not ' if n==-1 else '')+(pos if a==1 else neg);prefix=(f'Review: I thought the {noun} was ' if fmt=='thought' else f'Review: The {noun} was ');text=prefix+phrase+'. '+'Sentiment:';enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);gg=[[continuation_id(tok,text,s) for s in g] for g in groups('sentiment',['positive','negative'])];gg=[[v for v in g if v is not None] for g in gg]
    if any(not g for g in gg):excluded.append(dict(id=q['id'],a=a,n=n,reason='answer token boundary'));continue
    ast=len(prefix)+(4 if n==-1 else 0);end=len(prefix+phrase);adj=[t for t,(lo,hi) in enumerate(enc['offset_mapping']) if lo<end and hi>ast];period=next(t for t,(lo,hi) in enumerate(enc['offset_mapping']) if lo==end and hi==end+1);assert enc['input_ids'][period]==13 and adj
    q['indices'].append(len(ex));ex.append(dict(text=text,ids=enc['input_ids'],groups=gg,a=a,n=n,valence=a*n,adjective_position=adj[-1],adjective_token_count=len(adj),period_position=period,prior_prompt=text in prior_text))
   if len(q['indices'])==4:qs.append(q)
 assert len(ex)==192 and len(qs)==48 and not excluded and len({e['text'] for e in ex})==192
 manifest=ART/f'{S}_manifest.json';manifest.write_text(json.dumps(dict(examples=ex,quartets=qs,config=cfg),indent=2)+'\n');mh=sha(manifest);model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 indices=[];buf={}
 def take(x,role):return torch.stack([x[k,ex[i][role+'_position']] for k,i in enumerate(indices)]).detach().cpu()
 def mp(module,args):buf['normalized']=take(args[0],'adjective')
 def mo(module,args,out):buf['adjective_write']=take(out,'adjective')
 def po(module,args,out):buf['period_write']=take(out,'period')
 hooks=[model.blocks[3].mlp.register_forward_pre_hook(mp),model.blocks[3].mlp.register_forward_hook(mo),model.blocks[4].mlp.register_forward_hook(po)];native={};captures={};actual=0
 try:
  for length in sorted({len(e['ids']) for e in ex}):
   ii=[i for i,e in enumerate(ex) if len(e['ids'])==length]
   for st in range(0,len(ii),24):
    indices=ii[st:st+24];y,_=forward(model,torch.tensor([ex[i]['ids'] for i in indices],device='cuda'));actual+=len(indices)
    for k,i in enumerate(indices):native[i]=y[k].double().cpu();captures[i]={key:v[k].clone() for key,v in buf.items()}
 finally:
  for h in hooks:h.remove()
 records=[]
 for q in qs:
  mm=[measure(native[i],ex[i]['groups']) for i in q['indices']];correct=[m['margin']*ex[i]['valence']>0 for m,i in zip(mm,q['indices'])];full=[m['top1'][0 if ex[i]['valence']==1 else 1] for m,i in zip(mm,q['indices'])];records.append(q|dict(all_four_correct=all(correct),binary_correct=correct,full_vocab=sum(full)/4,margins=[m['margin'] for m in mm],single_token_adjectives=all(ex[i]['adjective_token_count']==1 for i in q['indices'])))
 summary=[]
 for fmt in ['film','book','story','thought']:
  for cohort in ['fresh1','fresh2','fresh3']:
   rr=[r for r in records if r['format']==fmt and r['cohort']==cohort];both=float(np.mean([r['all_four_correct'] for r in rr]));full=float(np.mean([r['full_vocab'] for r in rr]));summary.append(dict(format=fmt,cohort=cohort,n=len(rr),all_four_correct=both,full_vocab=full,state_binary_correct=np.mean([r['binary_correct'] for r in rr],axis=0).tolist(),single_token_quartets=sum(r['single_token_adjectives'] for r in rr),pass_gate=len(rr)>=4 and both>=.75 and full>=.5))
 check,_=forward(model,torch.tensor([ex[0]['ids']],device='cuda'));actual+=1;replay=float((check[0].double().cpu()-native[0]).abs().max());audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),manifest_unchanged=sha(manifest)==mh,whole_model_forwards=actual,expected_forwards=len(ex)+1,replay_logits_max_abs=replay,prior_prompt_overlap=sum(e['prior_prompt'] for e in ex),finite=all(np.isfinite(m) for r in records for m in r['margins']),no_training=True,test_evaluations=0)
 assert audits['hashes_unchanged'] and audits['manifest_unchanged'] and actual==len(ex)+1 and replay<1e-4 and audits['finite']
 torch.save(dict(examples=ex,quartets=qs,native=native,captures=captures),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,summary=summary,excluded=excluded,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(summary,audits,flush=True)
if __name__=='__main__':main()
