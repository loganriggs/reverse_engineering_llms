"""H798 native signed source-edge dose response with independent projection audit."""
import json,time,re,math
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_broad_task_competence import continuation_id
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
S='bilin18_native_edge_scaling'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'capture':ART/'bilin18_native_key_value_cross_captures.pt','reference':ART/'bilin18_native_key_value_cross.json','manifest':ART/'bilin18_native_key_value_cross_manifest.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};capture=torch.load(paths['capture'],weights_only=False);ref=json.loads(paths['reference'].read_text());manifest={r['id']:r for r in json.loads(paths['manifest'].read_text())};cfg={'experiment':'H798','hashes':hashes,'scales':[0.,.5,1.,1.5],'prediction':'Native edge scaling changes predictions; repeated-value competent cells test whether the negative edge affects intended-value probability. Do not infer output inhibition merely from routing sign.','design':'Scale only frozen L2H5 first-value→second-value edge before c_proj, leaving all other edges/heads intact. Independent residual-output subtraction check at scale0. Native KL, intended logprob, accuracy and pairwise margin separately.','limitations':'Native prompt, intervened internal computation. Same H797 panel and alignment exclusions. Different-key/different-value accuracy weak; conflicting-key cells have no unique correct answer.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={};edges={}
 for rid,c in capture.items():
  r=manifest[rid];enc=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True)
  spans=[m.span() for m in re.finditer(r'\b'+r['values'][0]+r'\b',r['text'])] if r['same_value'] else [re.search(r'\b'+v+r'\b',r['text']).span() for v in r['values']];positions=[]
  for a,b in spans:
   pp=[i for i,(x,y) in enumerate(enc['offset_mapping']) if x<b and y>a];assert len(pp)==1;positions.append(pp[0])
  ids=[continuation_id(tok,r['text'],' '+v) for v in r['contrast_values']];assert None not in ids;examples[rid]={**r,'ids':enc['input_ids'],'positions':positions,'answer_ids':ids};q,k,q2,k2,v=c['factors'].double();edges[rid]=(q@k/128)*(q2@k2/128)*v
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 buckets=defaultdict(list)
 for rid,e in examples.items():buckets[len(e['ids'])].append(rid)
 native={};forwards=0
 for rr in buckets.values():
  for lo in range(0,len(rr),12):
   bb=rr[lo:lo+12];out,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'));forwards+=len(bb)
   for i,k in enumerate(bb):native[k]=out[i].cpu()
 records=[];identity=[];zero_outputs={};reference=[];old={r['id']:r for r in ref['records']}
 for scale in cfg['scales']:
  for rr in buckets.values():
   for lo in range(0,len(rr),12):
    bb=rr[lo:lo+12]
    def patch(m,args):
     y=args[0].clone()
     for i,k in enumerate(bb):y[i,examples[k]['positions'][1],640:768]+=(scale-1)*edges[k].to(y)
     return(y,)
    h=model.blocks[2].attn.c_proj.register_forward_pre_hook(patch)
    try:out,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
    finally:h.remove()
    forwards+=len(bb)
    for i,k in enumerate(bb):
     e=examples[k];p=out[i].cpu();base=native[k];lp=p.double().log_softmax(0);blp=base.double().log_softmax(0);kl=float((blp.exp()*(blp-lp)).sum());a,b=e['answer_ids'];target=e['expected_index'];targetid=None if target is None else e['answer_ids'][target];r={n:e[n] for n in ['id','item','format','same_key','same_value','query','expected_index']};r.update(scale=scale,KL_native_to_scaled=kl,margin_change=float(p[a]-p[b]-base[a]+base[b]),intended_logprob_change=None if targetid is None else float(lp[targetid]-blp[targetid]),native_intended_prob=None if targetid is None else float(blp[targetid].exp()),scaled_intended_prob=None if targetid is None else float(lp[targetid].exp()),native_full_correct=None if targetid is None else int(base.argmax())==targetid,scaled_full_correct=None if targetid is None else int(p.argmax())==targetid,scaled_candidate_correct=None if targetid is None else int(p[e['answer_ids']].argmax())==target);records.append(r)
     if scale==1:identity.append(float((p-base).abs().max()));reference.append(abs(float(base[a]-base[b])-old[k]['candidate_margin']))
     if scale==0:zero_outputs[k]=p
 # Alternative intervention after projection: remove W_h * edge_h.
 W=model.blocks[2].attn.c_proj.weight[:,640:768].cpu().double();audit=[]
 for rr in buckets.values():
  for lo in range(0,len(rr),12):
   bb=rr[lo:lo+12]
   def alt(m,args,out):
    y=out[0].clone()
    for i,k in enumerate(bb):y[i,examples[k]['positions'][1]]-=(W@edges[k]).to(y)
    return(y,out[1])
   h=model.blocks[2].attn.register_forward_hook(alt)
   try:out,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:h.remove()
   forwards+=len(bb)
   for i,k in enumerate(bb):audit.append(float((out[i].cpu()-zero_outputs[k]).abs().max()))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'scale1_identity_logits_max_abs':max(identity),'H797_native_margin_replay_max_abs':max(reference),'pre_vs_post_projection_zero_edge_logits_max_abs':max(audit),'min_KL':min(r['KL_native_to_scaled'] for r in records),'finite':all(math.isfinite(r['KL_native_to_scaled']) for r in records),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(identity+reference+audit)<1e-4 and audits['min_KL']>-1e-10
 (ART/f'{S}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits)
if __name__=='__main__':main()
