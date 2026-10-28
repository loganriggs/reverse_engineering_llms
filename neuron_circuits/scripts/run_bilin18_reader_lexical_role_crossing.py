#!/usr/bin/env python3
"""H541: final-token, vocabulary and source-position matched reader tests."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,continuation_id,measure
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_reader_lexical_role_crossing';HEADS=[(14,4),(8,3),(14,0),(8,0)]
torch.set_num_threads(6)
def fresh():
 rows=[];names=[('Arthur','Olivia'),('Albert','Sophia'),('Fred','Maria'),('Philip','Nicole'),('Richard','Amy'),('Steven','Karen'),('Tony','Julie'),('Vincent','Angela')]
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 def pad(text,length):
  for n in range(64):
   candidate='Note '*n+text
   if len(tok.encode(candidate,add_special_tokens=False))==length:return candidate
  raise ValueError('Cannot achieve declared token length')
 for i,(a,b) in enumerate(names):
  standard=[f'{a} and {b} went to the market. {g} gave a parcel to' for g in (a,b)]
  target=len(tok.encode(standard[0],add_special_tokens=False))
  for fmt in ('standard','reversed_names','position_shift'):
   if fmt=='standard':prompts=standard
   elif fmt=='reversed_names':prompts=[f'{b} and {a} went to the market. {g} gave a parcel to' for g in (a,b)]
   else:prompts=[f'At the station, {a} met {b}. Later, {g} handed a ticket to' for g in (a,b)]
   rows.append(dict(id=f'matched-binding-{fmt}-{i}',family='entity_binding',format=fmt,split='fresh',prompts=prompts,answers=[b,a]))
  for fmt in ('standard','prefix','three_repeats','length_matched','position_matched'):
   if fmt=='standard':prompts=[f'to {v}; to {v}; to' for v in (a,b)]
   elif fmt=='prefix':prompts=[f'Sequence: to {v}; to {v}; to' for v in (a,b)]
   elif fmt=='three_repeats':prompts=[f'to {v}, to {v}, to {v}, to' for v in (a,b)]
   elif fmt=='length_matched':prompts=[pad(f'to {v}; to {v}; to',target) for v in (a,b)]
   else:prompts=[pad(f'to {v} and back; to {v} and back; to',target) for v in (a,b)]
   rows.append(dict(id=f'matched-copy-{fmt}-{i}',family='sequence_copy',format=fmt,split='fresh',prompts=prompts,answers=[a,b]))
 return rows
@torch.inference_mode()
def run(model,ids,capture=False,patch=None):
 d=model.config.n_embd;x0=F.rms_norm(model.embedding(ids),(d,));x=x0;bus=None;cache={}
 for l,b in enumerate(model.blocks):
  handles=[]
  if capture and l in (8,14):
   for n in ('c_q','c_k','c_q2','c_k2'):
    def hook(m,args,out,l=l,n=n):cache[(l,n)]=out.cpu()
    handles.append(getattr(b.attn,n).register_forward_hook(hook))
  if patch is not None and l==patch[0]:
   def ahook(m,args,out):
    y,v=out;y=y.clone();y[:,-1]+=patch[1].to(y);return y,v
   handles.append(b.attn.register_forward_hook(ahook))
  try:x,bus=b(x,bus,x0)
  finally:
   for h in handles:h.remove()
  if l==0 and capture:cache['bus']=bus.cpu()
 if capture:cache['last']=x[:,-1].cpu()
 return 30*torch.tanh(model.lm_head(F.rms_norm(x,(d,))[:,-1]).float()/30),cache
@torch.inference_mode()
def main():
 start=time.time();source=ART/'bilin18_reader_sign_confirmation.json';mp=ART/'bilin18_diverse_causal_tasks_manifest.json';old=[];rows=fresh()
 locked=json.loads((ART/'bilin18_reader_identity_transfer_locked_signs.json').read_text())
 third_name_pairs=[('Kevin','Brian'),('Ben','Zoe'),('Charles','Nancy'),('Edward','Helen'),('Harry','Betty'),('Louis','Anne'),('Martin','Joan'),('Walter','Carol')];third_pairs={'entity_binding':third_name_pairs,'sequence_copy':third_name_pairs}
 for row in rows:row['third_tokens']=third_pairs[row['family']][int(row['id'].rsplit('-',1)[1])]
 cfg={'experiment':'H541','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'heads':HEADS,'head_roles':['binding leader','copying leader','same-layer control','same-layer control'],'third_tokens':{'entity_binding':['Kevin','Brian'],'sequence_copy':['stone','glass']},'discovery':'Original H533 eight pairs per task only. Fit majority sign of unweighted token-identity map and routing coefficient separately, per task/head. Save signs before fresh analysis/intervention.','fresh_design':'Eight new lexical pairs per task x three formats; no new head selection. Native competence reported per format, all retained pairs included.','gate':'At least four pairs, >=75% both-endpoint candidate correctness, >=50% full-vocabulary top1.','decomposition':'Unweighted core = output_head @ (initial value_after-initial value_before). Identity alignment = (unembedding_after-unembedding_before) dot core. Coefficient = native attention pattern summed over changed source positions * bus_mix. Residual gain for frozen final effect reported separately.','causal_test':'Add the donor-minus-native persistent-bus contribution of one selected head at the last attention destination, then let all downstream computation run normally. Also inject each of two absent tokens into those source positions in this one head and compare their output margins.','third_test':'Output margin for third0 minus third1 under injection third0 minus that margin under injection third1. Positive means identity copying; negative means suppression. Neither third token is a declared task answer.','scope':'Head/source interventions, not replacement of all bus uses. No necessity or population claim.','precision':'float32 compute after bfloat16-rounded loader'}
 cfg.update(discovery='No fitting: H539 discovery identity-map signs frozen. Coefficient sign comes from each native prompt before intervention.', third_tokens=third_pairs, fresh_design='Eight different lexical pairs per task, three formats; eight third-token pairs per family. Some third-token words may have appeared previously in other roles.', gate_scales=[1], third_test='Predict sign(identity_map_sign_locked * native_coefficient * gate_scale) for third0-vs-third1 margin under third0 injection minus third1 injection. Zero gate is an exact invariance control.', causal_test='Replace the changed source-position bus contribution of one head by gate_scale times a third-token value. All downstream computation remains live. Gate sign reversal changes only this source-to-head bus contribution, not whole attention or QK.', sign_source_sha256=hashlib.sha256((ART/'bilin18_reader_identity_transfer_locked_signs.json').read_bytes()).hexdigest())
 cfg.update(fresh_design='Same eight name pairs and final token to across binding/copying. Binding anchors repeated from H540. Copying tests prefixes, repeats, matched final position, and matched final+nearest-source position. Last condition intervenes only on nearest changed source.', third_test='Only native gate +1; fixed-map sign times native coefficient predicts third-token contrast. No gate-reversal or zero-gate condition in this run.', matched_scope='For position_matched copying restrict bus intervention to last changed name position; this equals giver position in standard binding. Other copying formats use all repeated value positions.', hypothesis='Reader sign can differ across competent binding/copying contexts despite identical final token, names, third-token pairs and, in strongest condition, designated source/final positions.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');manifest=ART/f'{STEM}_manifest.json';manifest.write_text(json.dumps(rows,indent=2)+'\n');mh=hashlib.sha256(manifest.read_bytes()).hexdigest();print('LOCKED',len(rows),'pairs',flush=True)
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);model=load_full_model('cuda');assert model.config.squared_attn
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 def token_bus(token):
  x0=F.rms_norm(model.embedding(torch.tensor([token],device='cuda')),(model.config.n_embd,));x=model.blocks[0].lambdas.sum()*x0;x=F.rms_norm(x,(model.config.n_embd,));return model.blocks[0].attn.c_v(x).reshape(model.config.n_head,-1)
 examples={};excluded=[];native={};cache={};buckets=defaultdict(list)
 for r in rows:
  ee=[]
  for v,text in enumerate(r['prompts']):
   gs=[[continuation_id(tok,text,' '+a)] for a in r['answers']];third=[[continuation_id(tok,text,' '+a)] for a in r['third_tokens']];ee.append({'ids':tok.encode(text,add_special_tokens=False),'groups':gs,'third':third})
  if any(g[0] is None for e in ee for g in e['groups']+e['third']) or len(ee[0]['ids'])!=len(ee[1]['ids']):excluded.append(r['id']);continue
  for v,e in enumerate(ee):examples[(r['id'],v)]=e;buckets[len(e['ids'])].append((r['id'],v))
 for length,kk in sorted(buckets.items()):
  for begin in range(0,len(kk),8):
   keys=kk[begin:begin+8];logits,cc=run(model,torch.tensor([examples[k]['ids'] for k in keys],device='cuda'),capture=True)
   for ix,key in enumerate(keys):cache[key]={n:v[ix].clone() for n,v in cc.items()};native[key]=measure(logits[ix],examples[key]['groups'])
 torch.save({'cache':cache,'examples':examples},ART/f'{STEM}_native_cache.pt')
 result={'config':cfg,'manifest_sha256':mh,'excluded_pairs':excluded,'native':[dict(id=k[0],variant=k[1],**v) for k,v in native.items()],'geometry':[],'discovery_signs':locked,'competence':{},'interventions':[],'third_comparisons':[],'status':'running'};out=ART/f'{STEM}.json'
 def save():result['elapsed_seconds']=time.time()-start;out.write_text(json.dumps(result,indent=2)+'\n')
 jobs=defaultdict(list);buserr=[];factorerr=[];deltas={}
 # Analyze old discovery first and lock its signs before touching fresh geometry.
 for split in ('discovery','fresh'):
  for r in rows:
   if r['split']!=split or r['id'] in excluded:continue
   for rec,don in ((0,1),(1,0)):
    key=(r['id'],rec);dk=(r['id'],don);e=examples[key];c=cache[key];T=len(e['ids']);H=model.config.n_head;D=model.config.n_embd//H;positions=[i for i,(a,b) in enumerate(zip(e['ids'],examples[dk]['ids'])) if a!=b];delta=(cache[dk]['bus']-c['bus']).cuda();all_positions=positions.copy();positions=positions[-1:] if r['format']=='position_matched' else positions;mask=torch.zeros_like(delta);mask[positions]=delta[positions];delta=mask;before=e['ids'][positions[0]];after=examples[dk]['ids'][positions[0]]
    assert all(e['ids'][p]==before and examples[dk]['ids'][p]==after for p in positions)
    for token,actual in [(before,c['bus'][positions[0]]),(after,cache[dk]['bus'][positions[0]])]:buserr.append(float((token_bus(token).cpu()-actual).abs().max()))
    for l,h in HEADS:
     attn=model.blocks[l].attn;q,k,q2,k2=[c[(l,n)].cuda().view(1,T,H,D) for n in ('c_q','c_k','c_q2','c_k2')];cos,sin=attn.rotary(q)
     q,k=model.tt.apply_rotary_emb(F.rms_norm(q,(D,)),cos,sin),model.tt.apply_rotary_emb(F.rms_norm(k,(D,)),cos,sin);q2,k2=model.tt.apply_rotary_emb(F.rms_norm(q2,(D,)),cos,sin),model.tt.apply_rotary_emb(F.rms_norm(k2,(D,)),cos,sin)
     factor1=torch.einsum('bhd,bthd->bht',q[:,-1],k)/D;factor2=torch.einsum('bhd,bthd->bht',q2[:,-1],k2)/D;pattern=factor1*factor2;coeffs=pattern[0,h,positions]*attn.lamb;coefficient=coeffs.sum();w=attn.c_proj.weight[:,h*D:(h+1)*D];core=F.linear(delta[positions[0],h],w);identity=float((model.lm_head.weight[after]-model.lm_head.weight[before])@core);write=F.linear((pattern[0,h,:,None]*delta[:,h]*attn.lamb).sum(0),w);factorerr.append(float((write-coefficient*core).norm()/write.norm().clamp_min(1e-20)))
     gain=torch.ones((),device='cuda')
     for future in range(l+1,18):gain*=model.blocks[future].lambdas[0]
     rawsign=np.sign(identity*float(coefficient));predsign=rawsign*(-1 if r['family']=='entity_binding' else 1)
     record={'id':r['id'],'family':r['family'],'format':r['format'],'split':split,'recipient':rec,'layer':l,'head':h,'changed_positions':positions,'all_changed_positions':all_positions,'final_position':T-1,'final_token_id':e['ids'][-1],'factor1':factor1[0,h,positions].cpu().tolist(),'factor2':factor2[0,h,positions].cpu().tolist(),'identity_alignment':identity,'routing_coefficient':float(coefficient),'position_coefficients':coeffs.cpu().tolist(),'residual_gain':float(gain),'identity_effect_sign':float(rawsign),'predicted_task_direction_sign':float(predsign)};result['geometry'].append(record)
     delta_key=(r['id'],rec,l,h,'task');deltas[delta_key]=write.cpu();jobs[(l,T)].append((r,rec,l,h,'task',delta_key))
     if split=='fresh':
      for vi in (0,1):
       for gate in (1,):
        token=e['third'][vi][0];replacement=token_bus(token);vd=gate*replacement[h][None]-c['bus'][positions,h].cuda();ww=F.linear((coeffs[:,None]*vd).sum(0),w);mode=f'third{vi}_gate{gate}';dd=(r['id'],rec,l,h,mode);deltas[dd]=ww.cpu();jobs[(l,T)].append((r,rec,l,h,mode,dd))
    dd=(r['id'],rec,8,3,'identity');deltas[dd]=torch.zeros(model.config.n_embd);jobs[(8,T)].append((r,rec,8,3,'identity',dd))
  if split=='discovery':
   (ART/f'{STEM}_locked_signs.json').write_text(json.dumps(locked,indent=2)+'\n');print('EXISTING SIGNS LOCKED',flush=True)

 for fam in cfg['third_tokens']:
  result['competence'][fam]={}
  for fmt in sorted(set(r['format'] for r in rows if r['family']==fam)):
   rr=[r for r in rows if r['family']==fam and r['format']==fmt and r['id'] not in excluded];both=[native[(r['id'],0)]['margin']>0 and native[(r['id'],1)]['margin']<0 for r in rr];top=[native[(r['id'],v)]['top1'][v] for r in rr for v in (0,1)];n=len(rr);s={'pairs':n,'both_endpoint_choice':float(np.mean(both)) if n else 0,'full_vocab_top1':float(np.mean(top)) if n else 0};s['pass']=n>=4 and s['both_endpoint_choice']>=.75 and s['full_vocab_top1']>=.5;result['competence'][fam][fmt]=s
 save();print('SCREEN',result['competence'],flush=True)
 for (l,T),jj in sorted(jobs.items()):
  for begin in range(0,len(jj),8):
   batch=jj[begin:begin+8];ids=torch.tensor([examples[(r['id'],rec)]['ids'] for r,rec,l,h,mode,key in batch],device='cuda');ww=torch.stack([deltas[key] for r,rec,l,h,mode,key in batch]);logits,_=run(model,ids,patch=(l,ww))
   for ix,(r,rec,l,h,mode,dk) in enumerate(batch):
    key=(r['id'],rec);mm=measure(logits[ix],examples[key]['groups']);third=measure(logits[ix],examples[key]['third']);effect=(mm['margin']-native[key]['margin'])*(1 if rec==1 else -1);result['interventions'].append(dict(id=r['id'],family=r['family'],format=r['format'],split=r['split'],recipient=rec,layer=l,head=h,mode=mode,margin=mm['margin'],task_directed_delta=effect,third_margin=third['margin']))
  save()
 ix={(c['id'],c['recipient'],c['layer'],c['head'],c['mode']):c for c in result['interventions']}
 for g in result['geometry']:
  if g['split']!='fresh':continue
  key=(g['id'],g['recipient'],g['layer'],g['head']);map_sign=locked[g['family']][f"{g['layer']}:{g['head']}"]['identity_map_sign']
  for gate in (1,):
   effect=ix[key+(f'third0_gate{gate}',)]['third_margin']-ix[key+(f'third1_gate{gate}',)]['third_margin'];pred=float(np.sign(map_sign*g['routing_coefficient']*gate));result['third_comparisons'].append({**{k:g[k] for k in ('id','family','format','recipient','layer','head')},'gate':gate,'third_identity_effect':effect,'predicted_sign':pred,'sign_correct':bool(np.sign(effect)==pred) if gate else abs(effect)<1e-4})

 result['audits']={'token_bus_max_abs_replay_error':max(buserr),'coefficient_factorization_max_relative_l2_error':max(factorerr),'self_max_margin_delta':max(abs(c['task_directed_delta']) for c in result['interventions'] if c['mode']=='identity'),'finite':all(np.isfinite(c['margin']) and np.isfinite(c['third_margin']) for c in result['interventions']),'manifest_unchanged':mh==hashlib.sha256(manifest.read_bytes()).hexdigest(),'source_unchanged':cfg['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()};result['status']='complete';save();print('COMPLETE',flush=True)
if __name__=='__main__':main()
