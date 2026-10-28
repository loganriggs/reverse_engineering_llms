"""H778 exact routing/value/interaction head-edit decomposition and causal tests."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_l5_routing_value_decomposition'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'capture':ART/'bilin18_l5_copy_lexical_transfer_captures.pt','previous':ART/'bilin18_l5_copy_lexical_transfer.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','attention_source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py')};hashes={k:sha(p) for k,p in paths.items()};cap=torch.load(paths['capture'],weights_only=False);previous=json.loads(paths['previous'].read_text());examples=cap['examples'];native=cap['native_logits'];cfg={'experiment':'H778','hashes':hashes,'design':'At L5H5 capture actual post-RMS/RoPE q,k,q2,k2 and mixed values via squared_attention entry. Reconstruct signed A=(QK/128)*(Q2K2/128), causal mask, H=A V. Matched receiver/donor: routing=dA Vr;value=Ar dV;interaction=dA dV. Audit sum=Hd-Hr. Intervene each, routing+value, sum and full native head difference at query/source/both; reciprocal task,nuisance,identity across all H777lexicalcohorts.','limitations':'Receiver-centered algebra, not unique causal attribution; downstream nonlinear effects not additive. Routing is signed, not probabilities. Value includes layer-local and layer0bus mixture; no claim of purely lexical content or compact routing yet. Require equal sequence length and aligned selected source position.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 attn=model.blocks[5].attn;assert attn.squared_attn;original=attn.squared_attention;saved={};states={};buckets=defaultdict(list);forwards=0;nativeerr=[];recompose=[];headerr=[]
 for key,e in examples.items():buckets[len(e['ids'])].append(key)
 def intercept(q,k,v,q2,k2):
  A=(torch.einsum('bqhd,bkhd->bhqk',q,k)/128)*(torch.einsum('bqhd,bkhd->bhqk',q2,k2)/128);A.masked_fill_(~torch.ones(q.shape[1],q.shape[1],dtype=torch.bool,device=q.device).tril(),0);saved['A']=A[:,5].cpu();saved['V']=v[:,:,5].cpu();return original(q,k,v,q2,k2)
 attn.squared_attention=intercept
 def capture(m,args):saved['H']=args[0][:,:,640:768].cpu()
 hook=attn.c_proj.register_forward_pre_hook(capture)
 try:
  for length,kk in buckets.items():
   for lo in range(0,len(kk),12):
    bb=kk[lo:lo+12];logits,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'));forwards+=len(bb)
    for i,key in enumerate(bb):
     states[key]={n:saved[n][i].double() for n in ['A','V','H']};st=states[key];recompose.append(float((st['A']@st['V']-st['H']).abs().max()));nativeerr.append(float((logits[i].cpu()-native[key]).abs().max()));e=examples[key];headerr.append(float((st['H'][[length-1,e['source_position']]]-cap['native_heads'][key][:,640:768]).abs().max()))
 finally:attn.squared_attention=original;hook.remove()
 assert max(recompose+nativeerr+headerr)<1e-4
 torch.save({'states':states,'examples':examples},ART/f'{STEM}_captures.pt');jobs=[];components={};geometry=[];excluded=[];algebra=[]
 for rid in sorted({k[0] for k in examples}):
  for rv,dv,mode in [(0,1,'task'),(1,0,'task'),(0,2,'nuisance'),(0,0,'identity')]:
   key=(rid,rv);donor=(rid,dv);e=examples[key];de=examples[donor]
   if len(e['ids'])!=len(de['ids']) or e['source_position']!=de['source_position']:excluded.append({'id':rid,'receiver':rv,'donor':dv,'reason':'sequence/source alignment'});continue
   r=states[key];d=states[donor];dA=d['A']-r['A'];dV=d['V']-r['V'];terms={'routing':dA@r['V'],'value':r['A']@dV,'interaction':dA@dV};terms['routing_plus_value']=terms['routing']+terms['value'];terms['sum']=terms['routing_plus_value']+terms['interaction'];terms['full']=d['H']-r['H'];algebra.append(float((terms['sum']-terms['full']).abs().max()));components[key,donor]=terms
   for pos in ['query','source','both']:
    indices=[len(e['ids'])-1,e['source_position']] if pos=='both' else [len(e['ids'])-1 if pos=='query' else e['source_position']]
    if mode!='identity':geometry.append({'id':rid,'receiver':rv,'donor':dv,'mode':mode,'cohort':e['lexical_cohort'],'format':e['format'],'position':pos,'norms':{n:float(t[indices].norm()) for n,t in terms.items()},'routing_change_norm':float(dA[indices].norm()),'native_routing_norm':float(r['A'][indices].norm())})
    for term in terms:jobs.append({'key':key,'donor':donor,'mode':mode,'position':pos,'term':term,'indices':indices})
 assert max(algebra)<1e-4
 buckets=defaultdict(list)
 for j in jobs:buckets[len(examples[j['key']]['ids'])].append(j)
 records=[];identity=[]
 for length,jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12]
   def edit(m,args):
    y=args[0].clone()
    for i,j in enumerate(bb):
     for pos in j['indices']:y[i,pos,640:768]+=components[j['key'],j['donor']][j['term']][pos].to(y)
    return(y,)
   hook=attn.c_proj.register_forward_pre_hook(edit)
   try:out,_=run(model,torch.tensor([examples[j['key']]['ids'] for j in bb],device='cuda'))
   finally:hook.remove()
   forwards+=len(bb)
   for i,j in enumerate(bb):
    key=j['key'];e=examples[key];p=out[i].cpu();ln=native[key].double().log_softmax(-1);lp=p.double().log_softmax(-1);records.append({'id':key[0],'receiver':key[1],'donor':j['donor'][1],'mode':j['mode'],'position':j['position'],'term':j['term'],'cohort':e['lexical_cohort'],'format':e['format'],'margin_change':measure(p,e['groups'])['margin']-measure(native[key],e['groups'])['margin'],'native_to_edit_KL':float((ln.exp()*(ln-lp)).sum())})
    if j['mode']=='identity':identity.append(float((p-native[key]).abs().max()))
 refs={(r['id'],r['receiver'],r['donor'],r['position']):r for r in records if r['term']=='full'};oldindex={(r['id'],r['receiver'],r['donor'],r['position']):r for r in previous['records'] if r['variant']=='full'};replay=[abs(r['margin_change']-oldindex[k]['margin_change']) for k,r in refs.items()];sumreplay=[abs(r['margin_change']-refs[r['id'],r['receiver'],r['donor'],r['position']]['margin_change']) for r in records if r['term']=='sum'];groups=defaultdict(list)
 for r in records:
  if r['mode']!='identity':groups[r['cohort'],r['mode'],r['position'],r['term']].append(r)
 summary=[]
 for (cohort,mode,pos,term),rr in groups.items():
  ff=[refs[r['id'],r['receiver'],r['donor'],r['position']] for r in rr];rms=(sum(f['margin_change']**2 for f in ff)/len(ff))**.5;err=(sum((r['margin_change']-f['margin_change'])**2 for r,f in zip(rr,ff))/len(rr))**.5;summary.append({'cohort':cohort,'mode':mode,'position':pos,'term':term,'full_margin_RMS':rms,'component_margin_RMS':(sum(r['margin_change']**2 for r in rr)/len(rr))**.5,'absolute_RMS_error_to_full':err,'relative_RMS_error_to_full':err/rms if rms>=.25 else None})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_replay_max_abs':max(nativeerr),'cached_head_replay_max_abs':max(headerr),'AV_head_recomposition_max_abs':max(recompose),'component_sum_head_delta_max_abs':max(algebra),'full_margin_replay_max_abs':max(replay),'sum_full_margin_replay_max_abs':max(sumreplay),'identity_logit_replay_max_abs':max(identity),'finite':all(torch.isfinite(torch.tensor([r['margin_change'],r['native_to_edit_KL']])).all() for r in records),'aligned_edits':len(components),'excluded_edits':len(excluded),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(replay+sumreplay+identity)<1e-4
 (ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'value_mix_lambda':float(attn.lamb),'native_competence':previous['native_competence'],'excluded':excluded,'records':records,'geometry':geometry,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps(audits,indent=2),flush=True)
if __name__=='__main__':main()
