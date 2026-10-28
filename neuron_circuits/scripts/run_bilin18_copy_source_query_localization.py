"""H790 prespecified native MLP-output source/query substitutions on fresh copy panel."""
import json,time,re,math
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_copy_source_query_localization'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'manifest':ART/'bilin18_native_scale_replication_manifest.json','captures':ART/'bilin18_native_scale_replication_captures.pt','reference':ART/'bilin18_native_scale_replication.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cap=torch.load(paths['captures'],weights_only=False);ref=json.loads(paths['reference'].read_text());examples={k:e for k,e in cap['examples'].items() if e['task']=='sequence_copy'};native=cap['native'];manifest={r['id']:r for r in json.loads(paths['manifest'].read_text())};sites={'first_source':[0],'second_source':[1],'both_sources':[0,1],'query':[2],'sources_and_query':[0,1,2]};cfg={'experiment':'H790','hashes':hashes,'layers':[1,4],'sites':sites,'prediction':'Value-source native MLP-output swaps may have stronger donor-aligned effects than query swaps, distinguishing source representation from weak final-position choice. Test each format and each source individually; no post-outcome selection.','design':'Native donor output replacement at role-aligned token positions; both directions for task edits, answer-preserving key rename, identity. Whole downstream model live. Require equal token lengths and matching source/query token positions for paired interventions.','limitations':'Reused fresh H789 panel is now localization discovery. Joint sites change more tokens and often greater norm; no matched-energy dominance claim. Query is final token, which is punctuation in colon/newline format. No causal fractions or manifold claim.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);positions={};role_manifest=[]
 for k,e in examples.items():
  rid,v=k;r=manifest[rid];text=(r['prompts']+[r['nuisance']])[v];enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);assert enc['input_ids']==e['ids'];value=r['answers'][1 if v==1 else 0];spans=[m.span() for m in re.finditer(r'\b'+re.escape(value)+r'\b',text)];assert len(spans)==2;pp=[]
  for a,b in spans:
   ix=[i for i,(c,d) in enumerate(enc['offset_mapping']) if c<b and d>a];assert len(ix)==1;pp.append(ix[0])
  pp.append(len(e['ids'])-1);assert len(set(pp))==3;positions[k]=pp;role_manifest.append({'id':rid,'variant':v,'positions':pp,'final_token':tok.decode([e['ids'][-1]])})
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 writes={};saved={};batchkeys=[];handles=[];forwards=0;nativeerr=[];queryerr=[];queryrel=[]
 for l in [1,4]:
  def hook(m,args,out,l=l):saved[l]=torch.stack([out[i,positions[k]] for i,k in enumerate(batchkeys)]).cpu()
  handles.append(model.blocks[l].mlp.register_forward_hook(hook))
 buckets=defaultdict(list)
 for k,e in examples.items():buckets[len(e['ids'])].append(k)
 try:
  for kk in buckets.values():
   for lo in range(0,len(kk),12):
    batchkeys=kk[lo:lo+12];out,_=run(model,torch.tensor([examples[k]['ids'] for k in batchkeys],device='cuda'));forwards+=len(batchkeys)
    for i,k in enumerate(batchkeys):
     writes[k]={l:saved[l][i] for l in [1,4]};nativeerr.append(float((out[i].cpu()-native[k]).abs().max()));queryerr.extend(float((writes[k][l][2]-cap['outputs'][k][l]).abs().max()) for l in [1,4]);queryrel.extend(float((writes[k][l][2]-cap['outputs'][k][l]).norm()/cap['outputs'][k][l].norm().clamp_min(1e-12)) for l in [1,4])
 finally:
  for h in handles:h.remove()
 edits=[];excluded=[]
 for rid in sorted({k[0] for k in examples}):
  for rv,dv,mode in [(0,1,'task'),(1,0,'task'),(0,2,'nuisance'),(0,0,'identity')]:
   k=rid,rv;d=rid,dv
   if len(examples[k]['ids'])!=len(examples[d]['ids']) or positions[k]!=positions[d]:excluded.append({'id':rid,'receiver':rv,'donor':dv,'mode':mode,'reason':'token length/role position mismatch'});continue
   edits.append({'key':k,'donor':d,'mode':mode})
 jobs=[{**e,'layer':l,'site':site} for e in edits for l in [1,4] for site in sites];buckets=defaultdict(list)
 for job in jobs:buckets[job['layer'],len(examples[job['key']]['ids'])].append(job)
 records=[];identity=[];replaceerr=[];refidx={(r['id'],r['receiver'],r['donor'],r['layer']):r['margin_change'] for r in ref['records'] if r['control']=='native_output'};replay=[]
 for (l,length),jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12]
   def swap(m,args,out):
    y=out.clone()
    for i,j in enumerate(bb):
     k=j['key'];d=j['donor'];indices=sites[j['site']];pos=[positions[k][s] for s in indices];wanted=writes[d][l][indices].to(y);y[i,pos]=wanted;replaceerr.append(float((y[i,pos]-wanted).abs().max()))
    return y
   h=model.blocks[l].mlp.register_forward_hook(swap)
   try:out,_=run(model,torch.tensor([examples[j['key']]['ids'] for j in bb],device='cuda'))
   finally:h.remove()
   forwards+=len(bb)
   for i,j in enumerate(bb):
    k=j['key'];d=j['donor'];e=examples[k];p=out[i].cpu();base=measure(native[k],e['groups']);patched=measure(p,e['groups']);delta=patched['margin']-base['margin'];ix=sites[j['site']];change=writes[d][l][ix]-writes[k][l][ix];records.append({'id':k[0],'receiver':k[1],'donor':d[1],'mode':j['mode'],'family':e['family'],'item':e['item'],'layer':l,'site':j['site'],'margin_change':delta,'donor_directed_change':delta*(1 if k[1]==1 else -1) if j['mode']=='task' else None,'patched_margin':patched['margin'],'baseline_margin':base['margin'],'patch_squared_norm':float(change.square().sum()),'donor_choice':(patched['margin']>0 if d[1]==0 else patched['margin']<0) if j['mode']=='task' else None})
    if j['mode']=='identity':identity.append(float((p-native[k]).abs().max()))
    if j['site']=='query' and j['mode']!='identity':replay.append(abs(delta-refidx[k[0],k[1],d[1],l]))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_logits_replay_max_abs':max(nativeerr),'native_query_output_replay_max_abs':max(queryerr),'native_query_output_replay_max_relative_L2':max(queryrel),'query_swap_H789_margin_max_abs':max(replay),'output_replacement_max_abs':max(replaceerr),'identity_logits_max_abs':max(identity),'finite':all(math.isfinite(r['margin_change']) for r in records),'aligned_edits':len(edits),'excluded_edits':len(excluded),'whole_model_forwards':forwards,'test_evaluations':0};print(json.dumps(audits),flush=True);assert audits['hashes_unchanged'] and audits['finite'] and max(nativeerr+replay+replaceerr+identity)<1e-4 and max(queryrel)<1e-5
 torch.save({'writes':writes,'positions':positions},ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'role_manifest':role_manifest,'records':records,'excluded':excluded,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps(audits,indent=2));print('excluded',excluded)
if __name__=='__main__':main()
