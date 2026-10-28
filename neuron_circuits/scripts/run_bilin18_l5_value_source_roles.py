"""H780 explicit copied-value/key/query/separator roles for value edits."""
import json,time,re
from pathlib import Path
from collections import defaultdict
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_l5_value_source_roles'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'manifest':ART/'bilin18_l5_copy_lexical_transfer_manifest.json','streams':ART/'bilin18_l5_value_position_streams_captures.pt','native':ART/'bilin18_l5_copy_lexical_transfer_captures.pt','factors':ART/'bilin18_l5_routing_value_decomposition_captures.pt','previous':ART/'bilin18_l5_routing_value_decomposition.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cap=torch.load(paths['native'],weights_only=False);factors=torch.load(paths['factors'],weights_only=False)['states'];previous=json.loads(paths['previous'].read_text());examples=cap['examples'];native=cap['native_logits'];cfg={'experiment':'H780','hashes':hashes,'design':'Frozen-routing query output value edits split by first value occurrence, second value occurrence, non-final-token keys, final token, and remaining separators. Token offsets from original prompts determine roles; token ids checked against captured examples. Mixed and local-stream role edits, both value occurrences jointly, exact partition sums; no fitting. Same H778alignment gate.','limitations':'Final token can be punctuation in colon/equals formats; keys means all key tokens except final token. Contextual local values, not raw word identities. Source-role masks defined from prompt, not model outputs. No additive downstream causal fractions.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 attn=model.blocks[5].attn;lam=float(attn.lamb);streams=torch.load(paths['streams'],weights_only=False)['streams'];forwards=0;mixerr=[float(((1-lam)*st['local']+lam*st['bus']-factors[key]['V']).abs().max()) for key,st in streams.items()];assert max(mixerr)<1e-4
 manifest={r['id']:r for r in json.loads(paths['manifest'].read_text())};tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);roles={};rolemeta=[]
 for key,e in examples.items():
  rid,v=key;row=manifest[rid];text=row['prompts'][v];encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True);assert encoded['input_ids']==e['ids'];offsets=encoded['offset_mapping'];T=len(e['ids']);value=row['answers'][1 if v==1 else 0];value_spans=[m.span() for m in re.finditer(r'\b'+re.escape(value)+r'\b',text)];keyword=re.match(r'\w+',text).group();keyspans=[m.span() for m in re.finditer(r'\b'+re.escape(keyword)+r'\b',text)];assert len(value_spans)==2 and len(keyspans)==3
  def mask(spans):return torch.tensor([float(any(a<end and b>start for start,end in spans)) for a,b in offsets],dtype=torch.double)
  q=torch.zeros(T,dtype=torch.double);q[-1]=1;v1=mask(value_spans[:1]);v2=mask(value_spans[1:]);k=mask(keyspans)*(1-q);other=1-q-v1-v2-k;mm={'value_first':v1,'value_second':v2,'keys':k,'query':q,'separators':other};assert v1.sum()==1 and v2.sum()==1 and int(v2.nonzero()[0])==e['source_position'];assert all(bool((m>=0).all()) for m in mm.values()) and torch.equal(sum(mm.values()),torch.ones(T,dtype=torch.double));roles[key]=mm;rolemeta.append({'id':rid,'variant':v,'positions':{n:m.nonzero().flatten().tolist() for n,m in mm.items()}})
 terms={};geometry=[];excluded=[];algebra=[];jobs=[]
 for rid in sorted({k[0] for k in examples}):
  for rv,dv,mode in [(0,1,'task'),(1,0,'task'),(0,2,'nuisance'),(0,0,'identity')]:
   key=rid,rv;donor=rid,dv;e=examples[key];de=examples[donor];T=len(e['ids'])
   if T!=len(de['ids']) or e['source_position']!=de['source_position']:excluded.append({'id':rid,'receiver':rv,'donor':dv,'reason':'sequence/source alignment'});continue
   A=factors[key]['A'][-1];dV=factors[donor]['V']-factors[key]['V'];dl=(1-lam)*(streams[donor]['local']-streams[key]['local']);db=lam*(streams[donor]['bus']-streams[key]['bus']);masks=roles[key];assert all(torch.equal(masks[n],roles[donor][n]) for n in masks);t={'value':A@dV,'local':A@dl,'full_head':factors[donor]['H'][-1]-factors[key]['H'][-1]}
   for name,m in masks.items():t[name]=(A*m)@dV;t['local_'+name]=(A*m)@dl
   t['value_words']=t['value_first']+t['value_second'];t['local_value_words']=t['local_value_first']+t['local_value_second'];t['position_sum']=sum(t[n] for n in masks);t['local_position_sum']=sum(t['local_'+n] for n in masks);algebra.extend([float((t['value']-t['position_sum']).abs().max()),float((t['local']-t['local_position_sum']).abs().max())]);terms[key,donor]=t
   if mode!='identity':geometry.append({'id':rid,'receiver':rv,'donor':dv,'mode':mode,'cohort':e['lexical_cohort'],'format':e['format'],'norms':{n:float(v.norm()) for n,v in t.items()},'role_positions':{n:m.nonzero().flatten().tolist() for n,m in masks.items()}})
   for term in t:jobs.append({'key':key,'donor':donor,'mode':mode,'term':term})
 assert max(algebra)<1e-4
 buckets=defaultdict(list)
 for j in jobs:buckets[len(examples[j['key']]['ids'])].append(j)
 records=[];identity=[]
 for length,jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12];patch=torch.stack([terms[j['key'],j['donor']][j['term']] for j in bb]).cuda()
   def edit(m,args):y=args[0].clone();y[:,-1,640:768]+=patch.to(y);return(y,)
   hook=attn.c_proj.register_forward_pre_hook(edit)
   try:out,_=run(model,torch.tensor([examples[j['key']]['ids'] for j in bb],device='cuda'))
   finally:hook.remove()
   forwards+=len(bb)
   for i,j in enumerate(bb):
    key=j['key'];e=examples[key];p=out[i].cpu();records.append({'id':key[0],'receiver':key[1],'donor':j['donor'][1],'mode':j['mode'],'term':j['term'],'cohort':e['lexical_cohort'],'format':e['format'],'margin_change':measure(p,e['groups'])['margin']-measure(native[key],e['groups'])['margin']})
    if j['mode']=='identity':identity.append(float((p-native[key]).abs().max()))
 refs={(r['id'],r['receiver'],r['donor'],r['term']):r for r in records if r['term'] in ['value','local','full_head']};oldidx={(r['id'],r['receiver'],r['donor'],r['term']):r for r in previous['records'] if r['position']=='query'};replay=[];sumreplay=[];groups=defaultdict(list)
 for r in records:
  k=r['id'],r['receiver'],r['donor']
  if r['term'] in ['value','full_head']:replay.append(abs(r['margin_change']-oldidx[*k,'full' if r['term']=='full_head' else 'value']['margin_change']))
  if r['term'] in ['position_sum','local_position_sum']:sumreplay.append(abs(r['margin_change']-refs[*k,'local' if r['term']=='local_position_sum' else 'value']['margin_change']))
  if r['mode']!='identity':groups[r['cohort'],r['mode'],r['term']].append(r)
 summary=[]
 for (cohort,mode,term),rr in groups.items():
  row={'cohort':cohort,'mode':mode,'term':term,'component_margin_RMS':(sum(r['margin_change']**2 for r in rr)/len(rr))**.5}
  for ref in ['value','full_head']:
   ff=[refs[r['id'],r['receiver'],r['donor'],ref] for r in rr];rms=(sum(f['margin_change']**2 for f in ff)/len(ff))**.5;err=(sum((r['margin_change']-f['margin_change'])**2 for r,f in zip(rr,ff))/len(rr))**.5;row[ref+'_margin_RMS']=rms;row['absolute_RMS_error_to_'+ref]=err;row['relative_RMS_error_to_'+ref]=err/rms if rms>=.25 else None
  summary.append(row)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'mixed_value_reconstruction_max_abs':max(mixerr),'role_partition_examples':len(roles),'partition_vector_sum_max_abs':max(algebra),'H778_reference_margin_replay_max_abs':max(replay),'partition_sum_margin_replay_max_abs':max(sumreplay),'identity_logit_replay_max_abs':max(identity),'finite':all(torch.isfinite(torch.tensor(r['margin_change'])) for r in records),'aligned_edits':len(terms),'excluded_edits':len(excluded),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(replay+sumreplay+identity)<1e-4
 (ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'lambda':lam,'records':records,'summary':summary,'geometry':geometry,'role_manifest':rolemeta,'excluded':excluded,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps(audits,indent=2),flush=True)
if __name__=='__main__':main()
