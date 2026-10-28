"""H792 complete 4x4 factorial of native donor identities at L1 value sources."""
import json,time,math
from pathlib import Path
from collections import defaultdict
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
S='bilin18_source_identity_composition'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'captures':ART/'bilin18_source_donor_specificity_captures.pt','reference':ART/'bilin18_source_donor_specificity.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};c=torch.load(paths['captures'],weights_only=False);ref=json.loads(paths['reference'].read_text());examples=c['examples'];native=c['native'];writes=c['writes'];words=ref['candidate_words'];cfg={'experiment':'H792','hashes':hashes,'design':'Complete four-by-four factorial of first/second source identities (receiver0, donor1, donor2, donor3), native L1 MLP-output swaps. All16 combinations for every H791 item/format. Candidate logits/logprobs; additive prediction from single-source arms; coherent versus mixed composition.','prediction':'Independent content contributions predict joint candidate contrasts from single-source arms. A coherence-sensitive interaction yields systematic residual differences between equal and unequal donor identities. Such residuals do not by themselves identify an equality detector.','limitations':'Reused panel for compositional discovery, native output changes differ in norm; other residual paths remain receiver-native. Downstream nonlinearities can generate interactions without a discrete agreement gate.','primary_metric':'Logit interaction I(a,b)=delta_logits(a,b)-delta_logits(a,0)-delta_logits(0,b); evaluate mean target donor(s) contrast against other nonreceiver candidates.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 jobs=[{'id':rid,'a':a,'b':b} for rid in sorted({k[0] for k in examples}) for a in range(4) for b in range(4)];buckets=defaultdict(list)
 for j in jobs:buckets[len(examples[j['id'],0]['ids'])].append(j)
 records=[];identity=[];replacement=[];replay=[];forwards=0;old={(r['id'],r['donor_variant'],r['site']):r for r in ref['records']}
 for jj in buckets.values():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12]
   def patch(m,args,out):
    y=out.clone()
    for i,j in enumerate(bb):
     k=j['id'],0;pp=examples[k]['positions']
     for role,v in enumerate([j['a'],j['b']]):wanted=writes[j['id'],v][role].to(y);y[i,pp[role]]=wanted;replacement.append(float((y[i,pp[role]]-wanted).abs().max()))
    return y
   h=model.blocks[1].mlp.register_forward_hook(patch)
   try:out,_=run(model,torch.tensor([examples[j['id'],0]['ids'] for j in bb],device='cuda'))
   finally:h.remove()
   forwards+=len(bb)
   for i,j in enumerate(bb):
    k=j['id'],0;e=examples[k];p=out[i].cpu();base=native[k];ids=e['candidate_ids'];dl=(p-base)[ids];lp=p.log_softmax(0)[ids];blp=base.log_softmax(0)[ids];a,b=j['a'],j['b'];rr={'id':j['id'],'item':e['item'],'format':e['format'],'a':a,'b':b,'receiver_index':e['answer_index'],'variant_indices':[examples[j['id'],v]['answer_index'] for v in range(4)],'candidate_logit_changes':dl.tolist(),'candidate_logprob_changes':(lp-blp).tolist(),'candidate_logprobs':lp.tolist(),'candidate_prediction':int(p[ids].argmax()),'full_vocab_top_token':int(p.argmax()),'patch_squared_norm':float((writes[j['id'],a][0]-writes[k][0]).square().sum()+(writes[j['id'],b][1]-writes[k][1]).square().sum())};records.append(rr)
    if a==0 and b==0:identity.append(float((p-base).abs().max()))
    previous=old[j['id'],a,'both_sources'] if a==b else old[j['id'],a,'first_source'] if b==0 else old[j['id'],b,'second_source'] if a==0 else None
    if previous is not None:replay.append(float((dl-torch.tensor(previous['candidate_logit_changes'])).abs().max()))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'exact_output_replacement_max_abs':max(replacement),'identity_logits_max_abs':max(identity),'H791_candidate_delta_replay_max_abs':max(replay),'finite':all(math.isfinite(v) for r in records for v in r['candidate_logit_changes']),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(identity+replacement+replay)<1e-4;(ART/f'{S}.json').write_text(json.dumps({'status':'complete','config':cfg,'candidate_words':words,'competence':ref['competence'],'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits)
if __name__=='__main__':main()
