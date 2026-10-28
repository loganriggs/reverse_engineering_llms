#!/usr/bin/env python3
"""H538: executable fixed-reader formula and layer/head route transfer."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,groups,continuation_id,measure
from run_bilin18_bus_fixed_readers import run,MODES
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_bus_reader_decomposition'
torch.set_num_threads(6)
@torch.inference_mode()
def main():
 start=time.time();source=ART/'bilin18_bus_fixed_readers.json';prior=json.loads(source.read_text());mp=ART/'bilin18_diverse_causal_tasks_manifest.json';rows=[r for r in json.loads(mp.read_text()) if r['family'] in prior['config']['layers']]
 cfg={'experiment':'H538','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'layers':prior['config']['layers'],'formula':'Delta final residual = sum over future layers, heads, source positions: product of subsequent residual lambda0 * attention lamb * native final-query attention pattern * Down_head * Delta initial value_bus_head. Valid with all later QK, contextual V and MLP outputs fixed to recipient-native.','route_selection':'Aggregate each layer/head over all source positions. Rank by mean finite donor-directed contrast change on discovery, evaluated one route at a time. Freeze top1/3/5/10 routes and compare summed-route predictions on all transfer pairs. Report all routes and signed effects; no per-pair selection.','random_controls':'32 seeded uniformly chosen route sets for each size, independently per family.','seed':538202610,'scope':'Conditional fixed-reader surrogate, not native full-model replacement. Reused H533 panel. Layer/head routes differ from semantic axes.','hypotheses':['The fixed-reader formula reproduces H537 interventions.','A few fixed layer/head readers transfer across held-out names/colors and wording.'],'precision':'float32 compute after shared-loader bfloat16-rounded weights'}
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');print('LOCKED',flush=True)
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);model=load_full_model('cuda');assert model.config.squared_attn
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 examples={};cache={};native={};last={};buckets=defaultdict(list)
 for r in rows:
  for v,text in enumerate(r['prompts']):
   key=(r['id'],v);gs=[[continuation_id(tok,text,t) for t in g] for g in groups(r['family'],r['answers'])];assert all(len(g)==1 and g[0] is not None for g in gs);examples[key]={'ids':tok.encode(text,add_special_tokens=False),'groups':gs};buckets[len(examples[key]['ids'])].append(key)
 captured={}
 def finalhook(m,args,out):captured['last']=out[0][:,-1].detach().cpu().clone()
 hh=model.blocks[-1].register_forward_hook(finalhook)
 for length,kk in sorted(buckets.items()):
  for begin in range(0,len(kk),8):
   keys=kk[begin:begin+8];logits,cc=run(model,torch.tensor([examples[k]['ids'] for k in keys],device='cuda'),capture=True)
   for ix,key in enumerate(keys):cache[key]={n:v[ix].clone() for n,v in cc.items()};native[key]=measure(logits[ix],examples[key]['groups']);last[key]=captured['last'][ix].clone()
 def margin(x,gs):
  x=F.rms_norm(x.to('cuda'),(model.config.n_embd,));ww=model.lm_head.weight[[gs[0][0],gs[1][0]]];z=30*torch.tanh(F.linear(x,ww)/30);return z[...,0]-z[...,1]
 result={'config':cfg,'routes':[],'replay':[],'selection':{},'transfer':[],'status':'running'};out=ART/f'{STEM}.json';deltas={};edgeinfo=[]
 def save():result['elapsed_seconds']=time.time()-start;out.write_text(json.dumps(result,indent=2)+'\n')
 for r in rows:
  injection=cfg['layers'][r['family']]
  for rec,don in ((0,1),(1,0)):
   key=(r['id'],rec);dk=(r['id'],don);e=examples[key];c=cache[key];assert len(e['ids'])==len(examples[dk]['ids']);T=len(e['ids']);H=model.config.n_head;D=model.config.n_embd//H;delta=(cache[dk]['bus']-c['bus']).cuda();changed=[i for i,(x,y) in enumerate(zip(e['ids'],examples[dk]['ids'])) if x!=y];unchanged=[i for i in range(T) if i not in changed];unchangederr=float(delta[unchanged].abs().max()) if unchanged else 0.
   rr={}
   for l in range(injection+1,18):
    attn=model.blocks[l].attn
    q,k,q2,k2=[c[(l,n)].cuda().view(1,T,H,D) for n in ('c_q','c_k','c_q2','c_k2')];cos,sin=attn.rotary(q)
    q,k=model.tt.apply_rotary_emb(F.rms_norm(q,(D,)),cos,sin),model.tt.apply_rotary_emb(F.rms_norm(k,(D,)),cos,sin)
    q2,k2=model.tt.apply_rotary_emb(F.rms_norm(q2,(D,)),cos,sin),model.tt.apply_rotary_emb(F.rms_norm(k2,(D,)),cos,sin)
    pattern=(torch.einsum('bhd,bthd->bht',q[:,-1],k)/D)*(torch.einsum('bhd,bthd->bht',q2[:,-1],k2)/D)
    gain=attn.lamb.float()
    for future in range(l+1,18):gain=gain*model.blocks[future].lambdas[0].float()
    for h in range(H):
     # Last query can attend every source: its causal mask is all ones.
     weighted=pattern[0,h,:,None]*delta[:,h,:]*gain
     source_writes=F.linear(weighted,attn.c_proj.weight[:,h*D:(h+1)*D]);route=source_writes.sum(0).cpu();rr[f'{l}:{h}']=route
     for pos in changed:edgeinfo.append({'id':r['id'],'recipient':rec,'layer':l,'head':h,'position':pos,'source_before':tok.decode([e['ids'][pos]]),'source_after':tok.decode([examples[dk]['ids'][pos]]),'write_norm':float(source_writes[pos].norm())})
   delta_final=torch.stack(list(rr.values())).sum(0);frozen={(l,n):c[(l,n)][None] for l in range(injection+1,18) for n in MODES['fixed_qkv_mlp']};actual,_=run(model,torch.tensor([e['ids']],device='cuda'),injection=(injection,cache[dk]['bus'][None]),mode='fixed_qkv_mlp',frozen=frozen);actualx=captured['last'][0];pred=last[key]+delta_final;predmargin=float(margin(pred,e['groups']));actualmargin=measure(actual[0],e['groups'])['margin'];sign=1 if rec==1 else -1
   result['replay'].append(dict(id=r['id'],family=r['family'],split=r['split'],recipient=rec,changed_positions=changed,unchanged_bus_max_abs=unchangederr,hidden_max_abs_error=float((actualx-pred).abs().max()),hidden_relative_l2_error=float((actualx-pred).norm()/delta_final.norm()),margin_error=abs(predmargin-actualmargin),actual_margin=actualmargin,predicted_margin=predmargin,actual_directed_delta=(actualmargin-native[key]['margin'])*sign))
   route_names=list(rr);xs=torch.stack([last[key]+rr[n] for n in route_names]);effects=(margin(xs,e['groups']).cpu().numpy()-native[key]['margin'])*sign
   for n,effect in zip(route_names,effects):result['routes'].append(dict(id=r['id'],family=r['family'],split=r['split'],recipient=rec,route=n,directed_delta=float(effect),write_norm=float(rr[n].norm())))
   deltas[key]=rr
  print('decomposed',r['id'],flush=True)
 hh.remove();rng=np.random.default_rng(cfg['seed']);randomsets={}
 for fam in cfg['layers']:
  cc=[c for c in result['routes'] if c['family']==fam and c['split']=='discovery'];names=sorted(set(c['route'] for c in cc));scores={n:float(np.mean([c['directed_delta'] for c in cc if c['route']==n])) for n in names};ranked=sorted(names,key=lambda n:scores[n],reverse=True);result['selection'][fam]={'ranking':ranked,'discovery_scores':scores};randomsets[fam]={str(k):[rng.choice(names,k,replace=False).tolist() for _ in range(32)] for k in (1,3,5,10)}
 (ART/f'{STEM}_selected_routes.json').write_text(json.dumps({'selection':result['selection'],'random_sets':randomsets},indent=2)+'\n')
 for r in rows:
  if r['split']!='transfer':continue
  fam=r['family']
  for rec in (0,1):
   key=(r['id'],rec);rr=deltas[key];sets={'all':list(rr)}
   for k in (1,3,5,10):
    sets[f'top{k}']=result['selection'][fam]['ranking'][:k]
    for i,ss in enumerate(randomsets[fam][str(k)]):sets[f'random{k}_{i:02}']=ss
   xs=torch.stack([last[key]+torch.stack([rr[n] for n in names]).sum(0) for names in sets.values()]);effects=(margin(xs,examples[key]['groups']).cpu().numpy()-native[key]['margin'])*(1 if rec==1 else -1)
   for name,effect in zip(sets,effects):result['transfer'].append(dict(id=r['id'],family=fam,recipient=rec,method=name,directed_delta=float(effect)))
 # Store route vectors for independent replay, not full model activation caches.
 torch.save({'route_deltas':deltas,'native_final':last,'examples':examples},ART/f'{STEM}_route_vectors.pt');(ART/f'{STEM}_source_positions.json').write_text(json.dumps(edgeinfo,indent=2)+'\n')
 result['audits']={'hidden_max_abs_error':max(c['hidden_max_abs_error'] for c in result['replay']),'hidden_max_relative_l2_error':max(c['hidden_relative_l2_error'] for c in result['replay']),'margin_max_error':max(c['margin_error'] for c in result['replay']),'unchanged_bus_max_abs':max(c['unchanged_bus_max_abs'] for c in result['replay']),'finite':all(np.isfinite(c['directed_delta']) for c in result['routes']+result['transfer']),'manifest_unchanged':cfg['manifest_sha256']==hashlib.sha256(mp.read_bytes()).hexdigest(),'source_unchanged':cfg['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()};result['status']='complete';save();print('COMPLETE',flush=True)
if __name__=='__main__':main()
