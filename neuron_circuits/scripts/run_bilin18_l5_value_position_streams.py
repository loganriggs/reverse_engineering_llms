"""H779 local/bus and token-position partitions of frozen-routing value edits."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_l5_value_position_streams'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'native':ART/'bilin18_l5_copy_lexical_transfer_captures.pt','factors':ART/'bilin18_l5_routing_value_decomposition_captures.pt','previous':ART/'bilin18_l5_routing_value_decomposition.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cap=torch.load(paths['native'],weights_only=False);factors=torch.load(paths['factors'],weights_only=False)['states'];previous=json.loads(paths['previous'].read_text());examples=cap['examples'];native=cap['native_logits'];cfg={'experiment':'H779','hashes':hashes,'design':'Query-only decomposition of H778value term Ar dV. Partition sources into finalquery, changed-token-excluding-finalquery, allother positions (disjoint/exhaustive). Separately partition mixed values into(1-lambda)local+lambda layer0bus, and their6crossed position/stream terms. Compare each term to complete value edit and full head edit. Exact recombination audited; same alignment exclusions.','limitations':'Nuisance query can itself be a changed key token; finalquery takes precedence in disjoint masks. Local value includes upstream contextual processing, not raw embedding. Frozen receiver routing, additive vector identity not additive downstream attribution. Head output destination is final query only.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 attn=model.blocks[5].attn;lam=float(attn.lamb);saved={};streams={};forwards=0;nativeerr=[];mixerr=[];buckets=defaultdict(list)
 for key,e in examples.items():buckets[len(e['ids'])].append(key)
 def precapture(m,args):saved['bus']=args[1][:,:,5].cpu();saved['normalized_input']=args[0].cpu()
 def localcapture(m,args,out):saved['local']=out[:,:,640:768].cpu()
 hooks=[attn.register_forward_pre_hook(precapture),attn.c_v.register_forward_hook(localcapture)]
 try:
  for length,kk in buckets.items():
   for lo in range(0,len(kk),12):
    bb=kk[lo:lo+12];out,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'));forwards+=len(bb)
    for i,key in enumerate(bb):
     streams[key]={n:saved[n][i].double() for n in ['local','bus','normalized_input']};st=streams[key];mixerr.append(float(((1-lam)*st['local']+lam*st['bus']-factors[key]['V']).abs().max()));nativeerr.append(float((out[i].cpu()-native[key]).abs().max()))
 finally:
  for hook in hooks:hook.remove()
 assert max(mixerr+nativeerr)<1e-4
 torch.save({'streams':streams,'lambda':lam},ART/f'{STEM}_captures.pt');terms={};geometry=[];excluded=[];algebra=[];jobs=[]
 for rid in sorted({k[0] for k in examples}):
  for rv,dv,mode in [(0,1,'task'),(1,0,'task'),(0,2,'nuisance'),(0,0,'identity')]:
   key=rid,rv;donor=rid,dv;e=examples[key];de=examples[donor];T=len(e['ids'])
   if T!=len(de['ids']) or e['source_position']!=de['source_position']:excluded.append({'id':rid,'receiver':rv,'donor':dv,'reason':'sequence/source alignment'});continue
   A=factors[key]['A'][-1];dV=factors[donor]['V']-factors[key]['V'];dl=(1-lam)*(streams[donor]['local']-streams[key]['local']);db=lam*(streams[donor]['bus']-streams[key]['bus']);maskq=torch.zeros(T,dtype=torch.double);maskq[-1]=1;maskc=torch.tensor([int(a!=b) for a,b in zip(e['ids'],de['ids'])],dtype=torch.double);maskc[-1]=0;masko=1-maskq-maskc;assert torch.equal(maskq+maskc+masko,torch.ones(T,dtype=torch.double));masks={'changed':maskc,'query':maskq,'other':masko};t={'value':A@dV,'local':A@dl,'bus':A@db,'full_head':factors[donor]['H'][-1]-factors[key]['H'][-1]}
   for name,m in masks.items():
    t[name]=(A*m)@dV;t['local_'+name]=(A*m)@dl;t['bus_'+name]=(A*m)@db
   t['position_sum']=sum(t[n] for n in masks);t['stream_sum']=t['local']+t['bus'];algebra.extend([float((t['value']-t['position_sum']).abs().max()),float((t['value']-t['stream_sum']).abs().max())]);terms[key,donor]=t
   if mode!='identity':geometry.append({'id':rid,'receiver':rv,'donor':dv,'mode':mode,'cohort':e['lexical_cohort'],'format':e['format'],'norms':{n:float(v.norm()) for n,v in t.items()},'changed_positions':maskc.nonzero().flatten().tolist(),'query_position':T-1})
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
 refs={(r['id'],r['receiver'],r['donor'],r['term']):r for r in records if r['term'] in ['value','full_head']};oldidx={(r['id'],r['receiver'],r['donor'],r['term']):r for r in previous['records'] if r['position']=='query'};replay=[];sumreplay=[];groups=defaultdict(list)
 for r in records:
  k=r['id'],r['receiver'],r['donor']
  if r['term'] in ['value','full_head']:replay.append(abs(r['margin_change']-oldidx[*k,'full' if r['term']=='full_head' else 'value']['margin_change']))
  if r['term'] in ['position_sum','stream_sum']:sumreplay.append(abs(r['margin_change']-refs[*k,'value']['margin_change']))
  if r['mode']!='identity':groups[r['cohort'],r['mode'],r['term']].append(r)
 summary=[]
 for (cohort,mode,term),rr in groups.items():
  row={'cohort':cohort,'mode':mode,'term':term,'component_margin_RMS':(sum(r['margin_change']**2 for r in rr)/len(rr))**.5}
  for ref in ['value','full_head']:
   ff=[refs[r['id'],r['receiver'],r['donor'],ref] for r in rr];rms=(sum(f['margin_change']**2 for f in ff)/len(ff))**.5;err=(sum((r['margin_change']-f['margin_change'])**2 for r,f in zip(rr,ff))/len(rr))**.5;row[ref+'_margin_RMS']=rms;row['absolute_RMS_error_to_'+ref]=err;row['relative_RMS_error_to_'+ref]=err/rms if rms>=.25 else None
  summary.append(row)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'mixed_value_reconstruction_max_abs':max(mixerr),'native_replay_max_abs':max(nativeerr),'partition_vector_sum_max_abs':max(algebra),'H778_reference_margin_replay_max_abs':max(replay),'partition_sum_margin_replay_max_abs':max(sumreplay),'identity_logit_replay_max_abs':max(identity),'finite':all(torch.isfinite(torch.tensor(r['margin_change'])) for r in records),'aligned_edits':len(terms),'excluded_edits':len(excluded),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(replay+sumreplay+identity)<1e-4
 (ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'lambda':lam,'records':records,'summary':summary,'geometry':geometry,'excluded':excluded,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps(audits,indent=2),flush=True)
if __name__=='__main__':main()
