"""H741: exhaustive query/key-component crossing on matched operation contexts."""
import json,time,itertools
from pathlib import Path
from collections import defaultdict
import torch,numpy as np
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_static_bus_map import static_bus
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_recall_query_key_cross'
def coefficient(parts,lam):return lam*(parts['q1']@parts['k1']/128)*(parts['q2']@parts['k2']/128)
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_recall_operation_cross_captures.pt','previous':ART/'bilin18_recall_operation_cross.json','model':SNAPSHOT/'pytorch_model.bin','source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py'),'runner':Path(__file__),'static_bus':ROOT/'scripts/run_bilin18_static_bus_map.py','forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py'};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);previous=json.loads(paths['previous'].read_text());examples=data['examples'];native=data['native_logits'];oldgeo={(r['id'],r['receiver']):r for r in previous['geometry']};matches=previous['matching'];pairs=[]
 for m in matches:
  a=(f"cross-entity_binding-{m['format']}-{m['item']}",m['variant']);b=(f"cross-sequence_copy-{m['format']}-{m['item']}",m['variant']);pairs.extend([(a,b),(b,a)])
 cfg=dict(experiment='H741',hashes=hashes,design='64primaryH740contexts,32matchedbinding/copyingpairs bothdirections. CapturepostRMS/postRoPEdestinationq1,q2 anddesignatedsourcek1,k2 atL13H0. All16binaryrecipient/donor combinations of(q1,k1,q2,k2), bits0..3 respectively, withsameposition/token. Reconstructnativealpha andall-donoralpha. Replace onlydesignatedsource buscontribution byhybridalpha times originalsourcevalue orfixedabsenttokens; allotheredges/localvaluesnative. No QKswap acrossotheredges. Allhybridsreported.',controls='Native alpha, all-donor alpha, zero and reversednative coefficients. Native-value mode delta=(hybridalpha-nativealpha)*Bsource. Absentmode delta=hybridalpha*Babsent-nativealpha*Bsource. Threefixedabsentpairs fromH740; signprediction usesstaticmap, nofit. Zero/reverse controls applyabsentmode. Originalnative coefficient/valueidentity andH740absenteffectreplays.',limitations='These are surgical one-edge coefficient interventions, notreplacementofallqueries/keys or upstreamstates. Hybridstatesmaynotoccur naturally. Nativecompetence inheritedfromverifiedH740primarycells; reusedprompts. Full16crossingtests localization ofthis contrast, notabstractuniversaloperation semantics.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 attn=model.blocks[13].attn;lam=float(attn.lamb);W=attn.c_proj.weight[:,:128];U=model.lm_head.weight;cache={};native_errors=[];alpha_errors=[];forwards=0;buckets=defaultdict(list)
 for key,_ in pairs:buckets[len(examples[key]['ids'])].append(key)
 for length,keys in buckets.items():
  for lo in range(0,len(keys),12):
   bb=keys[lo:lo+12];cap={};hooks=[]
   for name in ['c_q','c_k','c_q2','c_k2']:
    def h(m,args,out,name=name):cap[name]=out[:,:,:128].cpu()
    hooks.append(getattr(attn,name).register_forward_hook(h))
   try:logits,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:
    for h in hooks:h.remove()
   forwards+=len(bb);raw=[cap[n].cuda()[:,:,None,:] for n in ['c_q','c_k','c_q2','c_k2']];cos,sin=attn.rotary(raw[0]);q,k,q2,k2=[model.tt.apply_rotary_emb(F.rms_norm(t,(128,)),cos,sin) for t in raw]
   for n,key in enumerate(bb):
    pos=examples[key]['source_position'];parts=dict(q1=q[n,-1,0].cpu(),k1=k[n,pos,0].cpu(),q2=q2[n,-1,0].cpu(),k2=k2[n,pos,0].cpu());cache[key]=parts;alpha_errors.append(abs(float(coefficient({s:v.double() for s,v in parts.items()},lam))-oldgeo[key]['alpha']));native_errors.append(float((logits[n].cpu()-native[key]).abs().max()))
 assert max(native_errors)<1e-4 and max(alpha_errors)<1e-5
 absentprior={(r['id'],r['receiver'],r['pair'],r['gate']):r for r in previous['absent']};geometry=[];jobs=[];meta={};replayalpha=[];order=['q1','k1','q2','k2']
 for key,donor in pairs:
  r={n:v.double() for n,v in cache[key].items()};d={n:v.double() for n,v in cache[donor].items()};alpha=float(coefficient(r,lam));target=float(coefficient(d,lam));e=examples[key];src=e['ids'][e['source_position']];Bsrc=static_bus(model,[src])[0];variants={mask:{n:(d[n] if mask&(1<<i) else r[n]) for i,n in enumerate(order)} for mask in range(16)};values={mask:float(coefficient(p,lam)) for mask,p in variants.items()};replayalpha.append(abs(values[15]-target));values['zero']=0.;values['reverse']=-alpha
  for mode,a in values.items():
   geometry.append(dict(id=key[0],receiver=key[1],family=e['family'],format=e['format'],donor_context=donor[0],mode=mode,native_alpha=alpha,donor_alpha=target,hybrid_alpha=a))
   if isinstance(mode,int):jobs.append(dict(key=key,mode=mode,kind='native_value',pair=None,variant=None,delta=((a-alpha)*Bsrc).cpu()))
   for pair in ['names','objects','countries']:
    prior=absentprior[key[0],key[1],pair,1];ids=prior['tokens'];vv=static_bus(model,ids);alignment=float((U[ids[0]]-U[ids[1]])@(W@(vv[0]-vv[1])));meta[key,mode,pair]=dict(id=key[0],receiver=key[1],family=e['family'],format=e['format'],mode=mode,pair=pair,alpha=a,predicted_alignment=a*alignment,tokens=ids)
    for v in [0,1]:jobs.append(dict(key=key,mode=mode,kind='absent',pair=pair,variant=v,delta=(a*vv[v]-alpha*Bsrc).cpu()))
 results={};effects=[];identity=[];buckets=defaultdict(list)
 for job in jobs:buckets[len(examples[job['key']]['ids'])].append(job)
 for length,jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12];delta=torch.stack([r['delta'] for r in bb]).cuda()
   def edit(m,args):
    y=args[0].clone();y[:,-1,:128]+=delta.to(y);return(y,)
   hook=attn.c_proj.register_forward_pre_hook(edit)
   try:logits,_=run(model,torch.tensor([examples[r['key']]['ids'] for r in bb],device='cuda'))
   finally:hook.remove()
   forwards+=len(bb)
   for n,job in enumerate(bb):
    key=job['key'];e=examples[key]
    if job['kind']=='absent':
     a,b=meta[key,job['mode'],job['pair']]['tokens'];results[key,job['mode'],job['pair'],job['variant']]=float(logits[n,a]-logits[n,b])
    else:
     change=measure(logits[n],e['groups'])['margin']-measure(native[key],e['groups'])['margin'];effects.append(dict(id=key[0],receiver=key[1],family=e['family'],format=e['format'],mode=job['mode'],intended_answer_margin_change=change*(1 if key[1]==0 else -1)))
     if job['mode']==0:identity.append(abs(change))
 absent=[];replay=[]
 for (key,mode,pair),m in meta.items():
  effect=results[key,mode,pair,0]-results[key,mode,pair,1];absent.append(dict(**m,observed_identity_effect=effect))
  if mode in [0,'zero','reverse']:gate={0:1,'zero':0,'reverse':-1}[mode];replay.append(abs(effect-absentprior[key[0],key[1],pair,gate]['observed_identity_effect']))
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},native_logits_replay_max_abs=max(native_errors),native_coefficient_replay_max_abs=max(alpha_errors),all_donor_coefficient_replay_max_abs=max(replayalpha),native_value_identity_margin_max_abs=max(identity),h740_absent_replay_max_abs=max(replay),zero_contrast_max_abs=max(abs(r['observed_identity_effect']) for r in absent if r['mode']=='zero'),finite=all(np.isfinite(r['observed_identity_effect']) for r in absent),whole_model_forwards=forwards,test_evaluations=0);print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(identity)<1e-4 and max(replay)<1e-4 and audits['zero_contrast_max_abs']<1e-4;torch.save(dict(qk=cache,matched_context_pairs=pairs),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,geometry=geometry,effects=effects,absent=absent,audits=audits,elapsed_seconds=time.time()-start),indent=2))
if __name__=='__main__':main()
