"""H791 frozen source-site confirmation with candidate-resolved readouts."""
import json,math,time,re
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_broad_task_competence import continuation_id
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_source_donor_specificity'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');keys='alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron omega'.split();values='train plane truck boat ship bike bus car taxi van jet cart tank sled coach trailer'.split();rows=[]
 for i in range(16):
  for f in [0,1]:
   vv=[values[(i+off)%16] for off in [0,3,7,11]];key=keys[i];prompts=[f'{key} {v} | {key} {v} | {key}' if f==0 else f'{key}: {v}\n{key}: {v}\n{key}:' for v in vv];rows.append({'id':f'h791-f{f}-{i:02d}','item':i,'format':f,'values':vv,'prompts':prompts})
 spec=ART/f'{STEM}_manifest.json';spec.write_text(json.dumps({'rows':rows,'candidate_words':values},indent=2)+'\n');cfg={'experiment':'H791','layer':1,'sites':{'first_source':[0],'second_source':[1],'both_sources':[0,1],'query':[2]},'prediction':'Native L1 source-output substitutions promote their actual donor identity more than other nonreceiver candidates, rather than only suppressing receiver confidence. Compare all three donor identities and single/both sources. Query is a frozen negative comparator.','design':'Fresh Greek-label/transport-value lexicons with H789 formats unchanged. Receiver variant0; donor variants1/2/3 and identity0. Full candidate logits/logprobs, donor-vs-receiver margin, and donor-specific contrast against other nonreceiver candidates. All candidates predetermined; no outcome-based filtering.','competence_gate':'Per format >=4 valid items, >=.75 all-four-variant candidate correctness, >=.5 pooled full-vocabulary intended top1.','limitations':'No matched replacement-energy control; joint sites affect more coordinates. Donor variants2/3 are alternative legitimate values for the same key, testing whether effects track actual donor. Values overlap across items; formats share items.','manifest_sha256':sha(spec),'runner_sha256':sha(Path(__file__)),'model_sha256':sha(SNAPSHOT/'pytorch_model.bin')};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={};positions={};excluded=[]
 for row in rows:
  ee=[]
  for v,text in enumerate(row['prompts']):
   ids=[continuation_id(tok,text,' '+word) for word in values]
   if any(t is None for t in ids) or len(set(ids))!=len(ids):break
   enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);spans=[m.span() for m in re.finditer(r'\b'+row['values'][v]+r'\b',text)];assert len(spans)==2;pp=[]
   for a,b in spans:
    ix=[i for i,(c,d) in enumerate(enc['offset_mapping']) if c<b and d>a]
    if len(ix)!=1:break
    pp.append(ix[0])
   if len(pp)!=2:break
   pp.append(len(enc['input_ids'])-1);ee.append({'ids':enc['input_ids'],'candidate_ids':ids,'positions':pp,'answer_index':values.index(row['values'][v]),'item':row['item'],'format':row['format']})
  if len(ee)!=4:excluded.append({'id':row['id'],'reason':'candidate or value-token validity'});continue
  if any(len(e['ids'])!=len(ee[0]['ids']) or e['positions']!=ee[0]['positions'] or e['candidate_ids']!=ee[0]['candidate_ids'] for e in ee):excluded.append({'id':row['id'],'reason':'alignment'});continue
  for v,e in enumerate(ee):examples[row['id'],v]=e;positions[row['id'],v]=e['positions']
 assert examples
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 native={};writes={};batchkeys=[];saved={};forwards=0
 def capture(m,args,out):saved['writes']=torch.stack([out[i,positions[k]] for i,k in enumerate(batchkeys)]).cpu()
 h=model.blocks[1].mlp.register_forward_hook(capture);buckets=defaultdict(list)
 for k,e in examples.items():buckets[len(e['ids'])].append(k)
 try:
  for kk in buckets.values():
   for lo in range(0,len(kk),12):
    batchkeys=kk[lo:lo+12];out,_=run(model,torch.tensor([examples[k]['ids'] for k in batchkeys],device='cuda'));forwards+=len(batchkeys)
    for i,k in enumerate(batchkeys):native[k]=out[i].cpu();writes[k]=saved['writes'][i]
 finally:h.remove()
 competence=[]
 for f in [0,1]:
  ids=sorted({k[0] for k,e in examples.items() if e['format']==f});allfour=[];top=[]
  for rid in ids:
   choices=[]
   for v in range(4):
    e=examples[rid,v];p=native[rid,v];target=e['candidate_ids'][e['answer_index']];choices.append(int(p[e['candidate_ids']].argmax())==e['answer_index']);top.append(int(p.argmax())==target)
   allfour.append(all(choices))
  mean=lambda a:sum(a)/len(a) if a else 0.;competence.append({'format':f,'items':len(ids),'all_four_candidate_correct':mean(allfour),'full_vocab_top1':mean(top),'pass':len(ids)>=4 and mean(allfour)>=.75 and mean(top)>=.5})
 print('competence',competence,'excluded',excluded,flush=True);jobs=[{'id':rid,'donor':dv,'site':site} for rid in sorted({k[0] for k in examples}) for dv in [0,1,2,3] for site in cfg['sites']];buckets=defaultdict(list)
 for j in jobs:buckets[len(examples[j['id'],0]['ids'])].append(j)
 records=[];identity=[];replacement=[];formula=[]
 for jj in buckets.values():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12]
   def patch(m,args,out):
    y=out.clone()
    for i,j in enumerate(bb):
     k=j['id'],0;d=j['id'],j['donor'];ix=cfg['sites'][j['site']];pp=[positions[k][z] for z in ix];wanted=writes[d][ix].to(y);y[i,pp]=wanted;replacement.append(float((y[i,pp]-wanted).abs().max()))
    return y
   h=model.blocks[1].mlp.register_forward_hook(patch)
   try:out,_=run(model,torch.tensor([examples[j['id'],0]['ids'] for j in bb],device='cuda'))
   finally:h.remove()
   forwards+=len(bb)
   for i,j in enumerate(bb):
    k=j['id'],0;dk=j['id'],j['donor'];e=examples[k];p=out[i].cpu();base=native[k];ids=e['candidate_ids'];a=e['answer_index'];b=examples[dk]['answer_index'];dl=(p-base)[ids];lp=p.log_softmax(0)[ids];blp=base.log_softmax(0)[ids];dlogp=lp-blp;other=[z for z in range(len(ids)) if z not in [a,b]];margin=float(dl[b]-dl[a]);formula.append(abs(margin-float(dlogp[b]-dlogp[a])));ix=cfg['sites'][j['site']];ref_b=examples[j['id'],1]['answer_index'];records.append({'id':j['id'],'item':e['item'],'format':e['format'],'site':j['site'],'donor_variant':j['donor'],'receiver_word':values[a],'donor_word':values[b],'candidate_logit_changes':dl.tolist(),'candidate_logprob_changes':dlogp.tolist(),'candidate_logprobs':lp.tolist(),'native_candidate_logprobs':blp.tolist(),'receiver_logprob_change':float(dlogp[a]),'donor_logprob_change':float(dlogp[b]),'donor_vs_receiver_margin_change':margin,'actual_donor_vs_other_alternatives_change':float(dl[b]-dl[other].mean()),'reference_alternate_vs_receiver_margin_change':float(dl[ref_b]-dl[a]),'donor_vs_reference_alternate_change':float(dl[b]-dl[ref_b]),'patch_squared_norm':float((writes[dk][ix]-writes[k][ix]).square().sum()),'candidate_prediction':values[int(p[ids].argmax())],'full_vocab_top_token':int(p.argmax()),'donor_full_vocab_top1':int(p.argmax())==ids[b],'native_receiver_prob':float(blp[a].exp()),'patched_receiver_prob':float(lp[a].exp()),'native_donor_prob':float(blp[b].exp()),'patched_donor_prob':float(lp[b].exp())})
    if j['donor']==0:identity.append(float((p-base).abs().max()))
 audits={'hashes_unchanged':sha(spec)==cfg['manifest_sha256'] and sha(Path(__file__))==cfg['runner_sha256'] and sha(SNAPSHOT/'pytorch_model.bin')==cfg['model_sha256'],'exact_output_replacement_max_abs':max(replacement),'identity_logits_max_abs':max(identity),'margin_logprob_identity_max_abs':max(formula),'finite':all(math.isfinite(r['donor_vs_receiver_margin_change']) for r in records),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(identity+replacement+formula)<1e-4
 torch.save({'native':native,'writes':writes,'examples':examples},ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'candidate_words':values,'competence':competence,'excluded':excluded,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps(audits,indent=2))
if __name__=='__main__':main()
