"""H794 causal additive clamps of early source/query factorial interactions."""
import json,time,math
from pathlib import Path
from collections import defaultdict
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
S='bilin18_early_interaction_removal'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'captures':ART/'bilin18_source_donor_specificity_captures.pt','reference':ART/'bilin18_source_identity_composition.json','early':ART/'bilin18_source_interaction_trace_early_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};c=torch.load(paths['captures'],weights_only=False);ref=json.loads(paths['reference'].read_text());examples=c['examples'];native=c['native'];writes=c['writes'];words=ref['candidate_words'];early=torch.load(paths['early'],weights_only=False);arms=['none','source_attention','query_mlp','both'];cfg={'experiment':'H794','hashes':hashes,'arms':arms,'design':'Full H792 factorial, L1 native source-output swaps. At L2 second-source attention output and/or query MLP output, clamp joint-state output to stored z_a0+z_0b-z_00. No clamp arm replays H792. All16factorial cells and all four arms retained.','prediction':'Removing second-source L2 attention interaction reduces final candidate interaction more than removing early query-local MLP interaction if the traced source route is relevant.','limitations':'Additive output clamps are hybrid causal tests, not unique mediation fractions. Joint arm clamps query MLP to original reference additive prediction, not a recomputed prediction after source clamp. Other token positions/paths live. Reused localization panel.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 jobs=[{'id':rid,'a':a,'b':b,'arm':arm} for rid in sorted({k[0] for k in examples}) for a in range(4) for b in range(4) for arm in arms];buckets=defaultdict(list)
 for j in jobs:buckets[len(examples[j['id'],0]['ids'])].append(j)
 records=[];identity=[];replacement=[];replay=[];forwards=0;old={(r['id'],r['a'],r['b']):r for r in ref['records']};clamp_errors=[]
 for jj in buckets.values():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12]
   def patch(m,args,out):
    y=out.clone()
    for i,j in enumerate(bb):
     k=j['id'],0;pp=examples[k]['positions']
     for role,v in enumerate([j['a'],j['b']]):wanted=writes[j['id'],v][role].to(y);y[i,pp[role]]=wanted;replacement.append(float((y[i,pp[role]]-wanted).abs().max()))
    return y
   def clamp_attention(m,args,out):
    y=out[0].clone()
    for i,j in enumerate(bb):
     if j['arm'] not in ['source_attention','both']:continue
     z=early[j['id']][2,'attn_output'].double().reshape(4,4,3,-1);desired=(z[j['a'],0,1]+z[0,j['b'],1]-z[0,0,1]).to(y);pos=examples[j['id'],0]['positions'][1];y[i,pos]=desired;clamp_errors.append(float((y[i,pos]-desired).abs().max()))
    return(y,out[1])
   def clamp_mlp(m,args,out):
    y=out.clone()
    for i,j in enumerate(bb):
     if j['arm'] not in ['query_mlp','both']:continue
     z=early[j['id']][2,'mlp_output'].double().reshape(4,4,3,-1);desired=(z[j['a'],0,2]+z[0,j['b'],2]-z[0,0,2]).to(y);y[i,-1]=desired;clamp_errors.append(float((y[i,-1]-desired).abs().max()))
    return y
   hs=[model.blocks[1].mlp.register_forward_hook(patch),model.blocks[2].attn.register_forward_hook(clamp_attention),model.blocks[2].mlp.register_forward_hook(clamp_mlp)]
   try:out,_=run(model,torch.tensor([examples[j['id'],0]['ids'] for j in bb],device='cuda'))
   finally:
    for h in hs:h.remove()
   forwards+=len(bb)
   for i,j in enumerate(bb):
    k=j['id'],0;e=examples[k];p=out[i].cpu();base=native[k];ids=e['candidate_ids'];dl=(p-base)[ids];lp=p.log_softmax(0)[ids];blp=base.log_softmax(0)[ids];a,b=j['a'],j['b'];rr={'id':j['id'],'item':e['item'],'format':e['format'],'a':a,'b':b,'arm':j['arm'],'receiver_index':e['answer_index'],'variant_indices':[examples[j['id'],v]['answer_index'] for v in range(4)],'candidate_logit_changes':dl.tolist(),'candidate_logprob_changes':(lp-blp).tolist(),'candidate_logprobs':lp.tolist(),'candidate_prediction':int(p[ids].argmax()),'full_vocab_top_token':int(p.argmax()),'patch_squared_norm':float((writes[j['id'],a][0]-writes[k][0]).square().sum()+(writes[j['id'],b][1]-writes[k][1]).square().sum())};records.append(rr)
    if a==0 and b==0:identity.append(float((p-base).abs().max()))
    if j['arm']=='none' or a==0 or b==0:replay.append(float((dl-torch.tensor(old[j['id'],a,b]['candidate_logit_changes'])).abs().max()))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'exact_output_replacement_max_abs':max(replacement),'additive_clamp_max_abs':max(clamp_errors),'identity_logits_max_abs':max(identity),'reference_or_single_cell_candidate_replay_max_abs':max(replay),'finite':all(math.isfinite(v) for r in records for v in r['candidate_logit_changes']),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(identity+replacement+replay)<1e-4;(ART/f'{S}.json').write_text(json.dumps({'status':'complete','config':cfg,'candidate_words':words,'competence':ref['competence'],'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits)
if __name__=='__main__':main()
