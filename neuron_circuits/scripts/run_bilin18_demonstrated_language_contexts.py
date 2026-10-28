"""H909 prospective language transfer competence/capture panel, fits frozen."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model,final_logits
from run_bilin18_broad_task_competence import continuation_id
from audit_bilin18_mlp_discriminators import TOKENIZER
from transformers import AutoTokenizer
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
S='bilin18_demonstrated_language_contexts'
NEW=[(['Netherlands','Austria'],['Dutch','German']),(['Brazil','Mexico'],['Portuguese','Spanish']),(['Egypt','Iran'],['Arabic','Persian']),(['Thailand','Vietnam'],['Thai','Vietnamese']),(['Ukraine','Romania'],['Ukrainian','Romanian']),(['Indonesia','Albania'],['Indonesian','Albanian']),(['Serbia','Croatia'],['Serbian','Croatian']),(['Czech Republic','Slovakia'],['Czech','Slovak'])]
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','old_panel':ART/'bilin18_multidomain_matched_competence.json',**{'frozen_mean_'+str(k):ART/f'bilin18_language_shared_partner_l{k}_fits.pt' for k in [12,13,14]}};hashes={k:sha(p) for k,p in paths.items()};old=[p for p in json.loads(paths['old_panel'].read_text())['config']['pairs'] if p['domain']=='language'];pairs=[dict(cohort='old',sources=p['sources'],answers=p['answers']) for p in old]+[dict(cohort='new',sources=a,answers=b) for a,b in NEW];cfg=dict(experiment='H914',hashes=hashes,pairs=pairs,formats=['N0','N1','N2'],demonstrations=[['Argentina','Spanish'],['Australia','English']],capture_layers=[13,14],gate=dict(min_pairs=4,min_both_binary=.75,min_top1=.5),design='Fixed H909 country inventory, common Input/Language/Country record format with zero/one/two demonstrations. Demonstration countries Argentina/Australia absent from evaluation. Exact same prefix across task/copy, one query token changes; counts/order fixed before outcomes. Oldpairs0–3 reserveddiscovery,4–7 oldtransfer, newcohort additionaltransfer. Samecountryinput, Language versus Country fieldquery. Verify equal tokenlength/finaltoken and changedtoken count before interpreting. Single continuation-token eligibility for task/copy answers, entirepair retained only if allconditionseligible. No correctness exclusions. Frozen H908means unchanged; baseline competence/captures only, no intervention outcomes. Allcountry identities usedearlier; prompt skeletons new to this investigation, not globally untouched/pretraining-disjoint.');(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples=[];excluded=[]
 for pair,p in enumerate(pairs):
  rows=[];errors=[]
  for fmt in cfg['formats']:
   for family in ['task','copy']:
    for order,country in enumerate(p['sources']):
     query='Language' if family=='task' else 'Country';prefix=''.join(f'Input: {c}\nLanguage: {l}\nCountry: {c}\n\n' for c,l in cfg['demonstrations'][:int(fmt[1:])]);text=prefix+f'Input: {country}\n{query}:';answers=p['answers'] if family=='task' else p['sources'];aid=[continuation_id(tok,text,' '+v) for v in answers]
     if any(v is None for v in aid):errors.append(dict(format=fmt,family=family,order=order,reason='answer_not_single'));continue
     rows.append(dict(pair=pair,cohort=p['cohort'],sources=p['sources'],answers=answers,format=fmt,family=family,order=order,expected=order,text=text,ids=tok.encode(text,add_special_tokens=False),answer_ids=aid))
  if errors:excluded.append(dict(pair=pair,sources=p['sources'],reasons=errors))
  else:examples.extend(rows)
 assert not {c for c,l in cfg['demonstrations']} & {c for p in pairs for c in p['sources']};assert examples;match={(e['pair'],e['format'],e['order'],e['family']):e for e in examples};token_audits=[]
 for e in examples:
  if e['family']!='task':continue
  c=match[e['pair'],e['format'],e['order'],'copy'];same_length=len(e['ids'])==len(c['ids']);changed=sum(a!=b for a,b in zip(e['ids'],c['ids'])) if same_length else None;token_audits.append(dict(pair=e['pair'],format=e['format'],order=e['order'],same_length=same_length,same_final_token=e['ids'][-1]==c['ids'][-1],changed_tokens=changed))
 assert all(r['same_length'] and r['same_final_token'] and r['changed_tokens']==1 for r in token_audits)
 (ART/f'{S}_manifest.json').write_text(json.dumps(dict(examples=examples,excluded=excluded,token_audits=token_audits),indent=2)+'\n');model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 stash={}
 def hook(k):
  def capture(mod,args,out):stash[k]=args[0][:,-1].double().cpu()
  return capture
 handles=[model.blocks[k].mlp.register_forward_hook(hook(k)) for k in cfg['capture_layers']];buckets=defaultdict(list)
 for i,e in enumerate(examples):buckets[len(e['ids'])].append(i)
 records=[];native={};states={};actual=0
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),8):
    ix=ii[st:st+8];y=final_logits(model,torch.tensor([examples[i]['ids'] for i in ix],device='cuda')).double().cpu();actual+=len(ix)
    for n,i in enumerate(ix):
     e=examples[i];ids=e['answer_ids'];sign=1 if e['order']==0 else -1;native[i]=y[n];states[i]={k:v[n].clone() for k,v in stash.items()};records.append(dict(index=i,pair=e['pair'],cohort=e['cohort'],format=e['format'],family=e['family'],order=e['order'],margin=sign*float(y[n,ids[0]]-y[n,ids[1]]),top1_correct=int(y[n].argmax())==ids[e['expected']]))
   print('DONE',length,actual,flush=True)
 finally:
  for h in handles:h.remove()
 cells=[]
 for cohort in ['old','new']:
  for fmt in cfg['formats']:
   for family in ['task','copy']:
    rr=[r for r in records if (r['cohort'],r['format'],r['family'])==(cohort,fmt,family)];pp=sorted({r['pair'] for r in rr});both=float(np.mean([all(v['margin']>0 for v in rr if v['pair']==p) for p in pp])) if pp else None;top=float(np.mean([r['top1_correct'] for r in rr])) if rr else None;cells.append(dict(cohort=cohort,format=fmt,family=family,pairs=len(pp),both_binary=both,top1=top,passes_gate=len(pp)>=4 and both>=.75 and top>=.5))
 audits=dict(token_matched_pairs=len(token_audits),all_pairs_equal_length_finaltoken_one_changed_token=all(r['same_length'] and r['same_final_token'] and r['changed_tokens']==1 for r in token_audits),hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),whole_model_forwards=actual,expected_forwards=len(examples),test_evaluations=0,finite=all(np.isfinite(r['margin']) for r in records));assert audits['hashes_unchanged'] and actual==len(examples) and audits['finite'];torch.save(dict(examples=examples,native=native,normalized_states=states),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,cells=cells,excluded=excluded,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(cells,flush=True);print(audits,flush=True)
if __name__=='__main__':main()
