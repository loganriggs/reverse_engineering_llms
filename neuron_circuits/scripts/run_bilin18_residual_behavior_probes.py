"""H999 fixed 16-family competence screen for expanded causal supervision."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import groups,measure,forward
from run_bilin18_broad_task_competence import continuation_id
from audit_bilin18_mlp_discriminators import TOKENIZER
from transformers import AutoTokenizer
from pathlib import Path
from collections import defaultdict
import torch,json,time,numpy as np
S='bilin18_residual_behavior_probes'
from bilin18_residual_probe_stimuli import suite
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'stimuli':ROOT/'scripts/bilin18_residual_probe_stimuli.py','tokenizer_helper':ROOT/'scripts/run_bilin18_broad_task_competence.py','forward_helper':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};rows=suite();assert len({tuple(r['prompts']) for r in rows})==len(rows), 'duplicate prompt pair'
 cfg={'experiment':'H999','hashes':hashes,'design':'Four candidate behaviors motivated by H998 natural residuals: clause completion, list markers, noun expectation, within-word fragments.8source pairs x2wordings. Native competence plus all18 last-token MLP captures; no featurizer fitting.','split':'A first4 discovery,last4 lexical; B first4 structural,last4 joint. These are development source slots; action vocabulary shared in noun probes. No native outcome-dependent replacements.','gate':'At least4distinct token-valid pairs, >=.75both-endpoint category correctness and >=.5intended full-vocabulary top1; report all failures/exclusions.','groups':'Fixed candidate token categories per family; continuation boundary checked independently for each endpoint; no overlapping categories. Word fragments use literal no-space BPE suffixes.','limits':'Matched current tokens checked, not assumed. List_marker quarantined diagnostic because related to original sequence_copy holdout. Other candidates require causal tests and confound controls before adding labels. No new-task discovery claim from competence alone.'}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');(ART/f'{S}_manifest.json').write_text(json.dumps(rows,indent=2)+'\n');mh=sha(ART/f'{S}_manifest.json');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);ex=[];pairs=[];excluded=[];seen={}
 for row in rows:
  texts=row['prompts'];groupstrings=row['group_strings'];ee=[]
  for variant,t in enumerate(texts):
   ids=tok.encode(t,add_special_tokens=False);gg=[[continuation_id(tok,t,s) for s in g] for g in groupstrings];gg=[[x for x in g if x is not None] for g in gg]
   if any(not g for g in gg) or set(gg[0])&set(gg[1]):break
   ee.append(dict(id=row['id'],variant=variant,text=t,ids=ids,groups=gg))
  if len(ee)!=2:excluded.append(dict(id=row['id'],reason='single-token answer boundary or overlapping groups'));continue
  key=tuple(texts);duplicate=key in seen;prior=seen.get(key);seen[key]=row['id'];ii=list(range(len(ex),len(ex)+2));pairs.append(row|dict(indices=ii,duplicate=duplicate,duplicate_of=prior,equal_length=len(ee[0]['ids'])==len(ee[1]['ids']),same_final_token=ee[0]['ids'][-1]==ee[1]['ids'][-1]));ex.extend(ee)
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 native={};writes={};actual=0
 for length in sorted({len(e['ids']) for e in ex}):
  ii=[i for i,e in enumerate(ex) if len(e['ids'])==length]
  for st in range(0,len(ii),24):
   inds=ii[st:st+24];ids=torch.tensor([ex[i]['ids'] for i in inds],device='cuda');y,cap=forward(model,ids,capture=True);actual+=len(inds)
   for k,i in enumerate(inds):
    native[i]=y[k].double().cpu();writes[i]=torch.stack([v[k] for v in cap])
 records=[]
 for row in pairs:
  measures=[measure(native[i],ex[i]['groups']) for i in row['indices']];records.append(row|dict(both_correct=measures[0]['margin']>0 and measures[1]['margin']<0,full_vocab=sum(measures[k]['top1'][k] for k in [0,1])/2,endpoint_metrics=measures))
 summary=[]
 for family in sorted({r['family'] for r in rows}):
  for split in ['discovery','lexical','structural','joint']:
   rr=[r for r in records if r['family']==family and r['split']==split];distinct={tuple(r['prompts']):r for r in rr};dd=list(distinct.values());both=float(np.mean([r['both_correct'] for r in dd])) if dd else 0;full=float(np.mean([r['full_vocab'] for r in dd])) if dd else 0
   summary.append(dict(family=family,split=split,same_final_token_pairs=sum(r['same_final_token'] for r in dd),declared_pairs=4,valid_pairs=len(rr),distinct_pairs=len(dd),both_correct=both,full_vocab=full,aligned_pairs=sum(r['equal_length'] and r['same_final_token'] for r in dd),pass_gate=len(dd)>=4 and both>=.75 and full>=.5,whole_task_holdout=next(r['whole_task_holdout'] for r in rows if r['family']==family)))
 # Independent repeat of the first token-valid example checks deterministic replay.
 check=ex[0];y,cap=forward(model,torch.tensor([check['ids']],device='cuda'),capture=True);actual+=1;replay=float((y[0].double().cpu()-native[0]).abs().max())
 capture_diff=torch.stack([v[0] for v in cap])-writes[0];capture_replay=float(capture_diff.abs().max());capture_relative=float(capture_diff.norm()/writes[0].norm());assert capture_relative<1e-5
 audits=dict(capture_replay_max_abs=capture_replay,capture_replay_relative=capture_relative,hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),manifest_unchanged=sha(ART/f'{S}_manifest.json')==mh,whole_model_forwards=actual,expected_forwards=len(ex)+1,replay_logits_max_abs=replay,finite=all(np.isfinite(m['margin']) for r in records for m in r['endpoint_metrics']),test_evaluations=0,no_training=True)
 assert audits['hashes_unchanged'] and audits['manifest_unchanged'] and audits['finite'] and actual==audits['expected_forwards'] and replay<1e-4
 torch.save(dict(examples=ex,native=native,writes=writes),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,summary=summary,records=records,excluded=excluded,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(summary,excluded,audits,flush=True)
if __name__=='__main__':main()
