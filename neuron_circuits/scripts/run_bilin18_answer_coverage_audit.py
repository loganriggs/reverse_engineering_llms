"""H760 symmetric answer-format coverage audit of cached native logits."""
import json,time
from pathlib import Path
from collections import defaultdict,Counter
import torch
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART,groups,measure
from run_bilin18_broad_task_competence import continuation_id
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_answer_coverage_audit'
def main():
 torch.set_num_threads(6);paths={'captures':ART/'bilin18_fresh_head_tasks_captures.pt','manifest':ART/'bilin18_fresh_head_tasks_manifest.json','results':ART/'bilin18_fresh_head_tasks.json','runner':Path(__file__)};hashes={k:sha(v) for k,v in paths.items()};cfg={'experiment':'H760','hashes':hashes,'design':'Cached H732 native logits, all valid sentiment and capital-recall endpoints. Original answer sets versus symmetric surface-form expansions: canonical labels with lower/title/upper case and zero/one leading space; expanded existing semantic set with same case/space variants. Only exact single-token continuations with unchanged prompt tokenization retained; deduplicate token IDs and enforce class disjointness. Rules fixed before reading predictions in this audit. Report original replay, contrast/top1, candidate mass/ranks and native top5. No selected-case filtering.','limitations':'Posthoc measurement audit of previously inspected failures; alternative scoring is diagnostic, not retroactive validation of old causal metrics. Multi-token answers or delayed whitespace/newline answers are not scored here.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);cap=torch.load(paths['captures'],weights_only=False);manifest=json.loads(paths['manifest'].read_text());old=json.loads(paths['results'].read_text());lookup={r['id']:r for r in manifest};records=[];excluded=[];overlap=[]
 for key,e in cap['examples'].items():
  rid,v=key
  if e['family'] not in ['sentiment','capital_recall'] or v not in [0,1]:continue
  row=lookup[rid];prompt=row['prompts'][v];logits=cap['native_logits'][key];original=groups(e['family'],row['answers']);methods={'original':original}
  for name,words in [('canonical_formats',[[a] for a in row['answers']]),('expanded_semantic_formats',original)]:
   methods[name]=[sorted({prefix+form for word in label for form in [word.strip().lower(),word.strip().capitalize(),word.strip().upper()] for prefix in ['', ' ']}) for label in words]
  for name,candidates in methods.items():
   gs=[]
   for c,strings in enumerate(candidates):
    ids=[]
    for value in strings:
     t=continuation_id(tok,prompt,value)
     if t is None:excluded.append({'id':rid,'variant':v,'method':name,'class':c,'candidate':value})
     else:ids.append(t)
    gs.append(sorted(set(ids)))
   assert all(gs);overlap.append(len(set(gs[0])&set(gs[1])));assert not overlap[-1]
   vals,idx=logits.topk(5);records.append({'id':rid,'variant':v,'family':e['family'],'format':e['split'],'method':name,'candidate_ids':gs,'candidate_text':[[tok.decode([t]) for t in g] for g in gs],'candidate_counts':[len(g) for g in gs],'top5':[{'text':tok.decode([int(t)]),'logit':float(x)} for t,x in zip(idx,vals)],**measure(logits,gs)})
 summaries=[];replay=[]
 for family in ['sentiment','capital_recall']:
  for fmt in ['format_a','format_b']:
   for method in ['original','canonical_formats','expanded_semantic_formats']:
    rr=[r for r in records if r['family']==family and r['format']==fmt and r['method']==method];ix={(r['id'],r['variant']):r for r in rr};ids=sorted({r['id'] for r in rr});both=sum(ix[rid,0]['margin']>0 and ix[rid,1]['margin']<0 for rid in ids)/len(ids);top=sum(r['top1'][r['variant']] for r in rr)/len(rr);s={'family':family,'format':fmt,'method':method,'pairs':len(ids),'both_endpoint_choice':both,'full_vocab_top1':top,'pass':len(ids)>=4 and both>=.75 and top>=.5,'mean_correct_mass':sum(r['mass'][r['variant']] for r in rr)/len(rr),'native_top1_counts':dict(Counter(r['top5'][0]['text'] for r in rr))};summaries.append(s)
    if method=='original':prev=old['summary'][family][fmt];replay.extend([abs(prev['both_endpoint_choice']-both),abs(prev['full_vocab_top1']-top)])
 audits={'hashes_unchanged':{k:sha(v)==hashes[k] for k,v in paths.items()},'original_competence_replay_max_abs':max(replay),'class_token_overlap_max':max(overlap),'whole_model_forwards':0,'test_evaluations':0};assert all(audits['hashes_unchanged'].values()) and max(replay)==0
 (ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'summary':summaries,'excluded_non_single_token_forms':excluded,'audits':audits},indent=2)+'\n');print(json.dumps({'summary':summaries,'audits':audits},indent=2))
if __name__=='__main__':main()
