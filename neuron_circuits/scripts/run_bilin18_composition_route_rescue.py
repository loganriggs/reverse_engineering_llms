"""H974 downstream rescue after adjective MLP3 factorial ablation."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward,measure
from pathlib import Path
from collections import defaultdict
import torch,json,numpy as np,time
S='bilin18_composition_route_rescue'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'source':ART/'bilin18_source_position_composition_captures.pt','attention':ART/'bilin18_attention5_sources_captures.pt','trace':ART/'bilin18_valence_interaction_trace_captures.pt','data':ART/'bilin18_factorial_valence_negation_captures.pt','panel':ART/'bilin18_factorial_valence_negation.json','model':SNAPSHOT/'pytorch_model.bin','forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py'};hashes={k:sha(p) for k,p in paths.items()};src=torch.load(paths['source'],weights_only=False);attdata=torch.load(paths['attention'],weights_only=False);trace=torch.load(paths['trace'],weights_only=False);data=torch.load(paths['data'],weights_only=False);panel=json.loads(paths['panel'].read_text());ex=data['examples'];qs=data['quartets'];native=data['native'];pos=src['positions'];W=attdata['projection'];native_sources={}
 for i,e in enumerate(ex):
  p=attdata['captures'][i]['P'].double();v=attdata['captures'][i]['V'].double();z=torch.einsum('ht,thd,ohd->to',p,v,W);adj=[t for t,s in enumerate(attdata['slots'][i]) if s.startswith('adjective:')];period=attdata['slots'][i].index('suffix:0');native_sources[i]={'attn_adjective':z[adj].sum(0).float(),'attn_period':z[period].float()};native_sources[i]['attn_both']=native_sources[i]['attn_adjective']+native_sources[i]['attn_period']
 cfg={'experiment':'H974','hashes':hashes,'design':'After local or total adjective-position MLP3 interaction removal, restore native period MLP4 output, or native adjective/period/both source contributions to attention5 final write, or full attention5 final write. Source rescue adds native contribution minus currently perturbed contribution, retaining other sources and native attention bus. Each rescue has paired random delta controls norm-matched to its actual correction.','native_controls':'Run every rescue on unablated prompts as an identity check, with matched nearzero random deltas. Include each ablation without rescue.','metrics':'Native-reference KL and intended-margin loss; fraction of ablation margin loss recovered and KL remaining relative to no-rescue ablation. These are conditional intervention effects, not unique mediation percentages or additive path fractions.','scope':'Locked panel; only four familiar original-format quartets pass native gate. Whole-source and whole-write rescues restore more than just factorial interaction, so interpret as route capacity, not identification of an exclusive mechanism. No training or reserved test.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 from jacclust.tt_model import apply_rotary_emb
 current=[];buf={};normerr=0.;reconstruction=0.;records=[];actual=0;native_control_err=0.
 def ablate(module,args,out):
  y=out.clone()
  for k,j in enumerate(current):
   if j['ablation']=='none':continue
   pp=src['parts'][j['q']['id'],'adjective',3];c=pp['local'] if j['ablation']=='local' else pp['local']+pp['inherited'];i=j['i'];y[k,pos[i]['adjective']]=src['captures'][i][3,'mlp'][0].to(y)-ex[i]['valence']*c.to(y)
  return y
 def correction(delta,j):
  nonlocal normerr
  if j['arm']=='restore':return delta
  gen=torch.Generator().manual_seed(974000+j['i']*20+['period_mlp','attn_adjective','attn_period','attn_both','attn_full'].index(j['rescue']));z=torch.randn(1152,generator=gen).to(delta);z=z/z.norm()*delta.norm();normerr=max(normerr,float(abs(z.norm()-delta.norm())/delta.norm().clamp_min(1e-20)));return z if j['arm']=='random_plus' else -z
 def periodhook(module,args,out):
  y=out.clone()
  for k,j in enumerate(current):
   if j['rescue']!='period_mlp':continue
   i=j['i'];t=pos[i]['period'];delta=src['captures'][i][4,'mlp'][1].to(y)-out[k,t];y[k,t]+=correction(delta,j)
  return y
 def attpre(module,args):
  x,bus=args;b,t,c=x.shape;h=module.n_head;dd=module.head_dim;q=module.c_q(x).view(b,t,h,dd);k=module.c_k(x).view_as(q);q2=module.c_q2(x).view_as(q);k2=module.c_k2(x).view_as(q);v=(1-module.lamb)*module.c_v(x).view_as(q)+module.lamb*bus.view_as(q);cos,sin=module.rotary(q)
  q=apply_rotary_emb(torch.nn.functional.rms_norm(q,(dd,)),cos,sin);k=apply_rotary_emb(torch.nn.functional.rms_norm(k,(dd,)),cos,sin);q2=apply_rotary_emb(torch.nn.functional.rms_norm(q2,(dd,)),cos,sin);k2=apply_rotary_emb(torch.nn.functional.rms_norm(k2,(dd,)),cos,sin);p=(torch.einsum('bhd,bthd->bht',q[:,-1],k)/dd)*(torch.einsum('bhd,bthd->bht',q2[:,-1],k2)/dd);w=module.c_proj.weight.view(c,h,dd);buf['z']=torch.einsum('bht,bthd,ohd->bto',p,v,w)
 def atthook(module,args,out):
  nonlocal reconstruction
  y,bus=out;z=buf['z'];reconstruction=max(reconstruction,float((z.sum(1)-y[:,-1]).norm()/y[:,-1].norm()));edited=y.clone()
  for k,j in enumerate(current):
   rescue=j['rescue'];i=j['i']
   if not rescue.startswith('attn_'):continue
   if rescue=='attn_full':delta=trace['captures'][i][5,'attention'].to(y)-y[k,-1]
   else:
    adj=[t for t,s in enumerate(attdata['slots'][i]) if s.startswith('adjective:')];period=pos[i]['period'];old=z[k,adj].sum(0) if rescue=='attn_adjective' else (z[k,period] if rescue=='attn_period' else z[k,adj].sum(0)+z[k,period]);delta=native_sources[i][rescue].to(y)-old
   edited[k,-1]+=correction(delta,j)
  return edited,bus
 handles=[model.blocks[3].mlp.register_forward_hook(ablate),model.blocks[4].mlp.register_forward_hook(periodhook),model.blocks[5].attn.register_forward_pre_hook(attpre),model.blocks[5].attn.register_forward_hook(atthook)]
 jobs=[]
 for ablation in ['none','local','total']:
  for q in qs:
   for state,i in enumerate(q['indices']):
    b=dict(ablation=ablation,q=q,state=state,i=i);jobs.append(b|dict(rescue='none',arm='none'))
    for rescue in ['period_mlp','attn_adjective','attn_period','attn_both','attn_full']:
     for arm in ['restore','random_plus','random_minus']:jobs.append(b|dict(rescue=rescue,arm=arm))
 buckets=defaultdict(list)
 for j in jobs:buckets[len(ex[j['i']]['ids'])].append(j)
 try:
  for length,jj in sorted(buckets.items()):
   for st in range(0,len(jj),24):
    current=jj[st:st+24];out,_=forward(model,torch.tensor([ex[j['i']]['ids'] for j in current],device='cuda'));actual+=len(current)
    for k,j in enumerate(current):
     y=out[k].double().cpu();base=native[j['i']];lp=y.log_softmax(0);ln=base.log_softmax(0);m=measure(y,ex[j['i']]['groups'])['margin'];bm=measure(base,ex[j['i']]['groups'])['margin'];v=ex[j['i']]['valence']
     if j['ablation']=='none':native_control_err=max(native_control_err,float((y-base).abs().max()))
     records.append({key:j['q'][key] for key in ['id','format','cohort','source']}|dict(state=j['state'],ablation=j['ablation'],rescue=j['rescue'],arm=j['arm'],margin=m,loss=v*(bm-m),KL=float((ln.exp()*(ln-lp)).sum()),binary_correct=v*m>0))
   print('DONE',length,actual,flush=True)
 finally:
  for h in handles:h.remove()
 grouped=defaultdict(list)
 for r in records:grouped[r['format'],r['cohort'],r['ablation'],r['rescue'],r['arm']].append(r)
 summary=[]
 for key,rr in grouped.items():
  ref=grouped[key[0],key[1],key[2],'none','none'];loss=float(np.mean([r['loss'] for r in rr]));base_loss=float(np.mean([r['loss'] for r in ref]));kl=float(np.mean([r['KL'] for r in rr]));basekl=float(np.mean([r['KL'] for r in ref]));summary.append(dict(zip(['format','cohort','ablation','rescue','arm'],key))|dict(n=len(rr),mean_loss=loss,mean_KL=kl,margin_recovery=(base_loss-loss)/base_loss if key[2]!='none' else None,KL_remaining=kl/basekl if key[2]!='none' else None,binary_correct=float(np.mean([r['binary_correct'] for r in rr]))))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),whole_model_forwards=actual,expected_forwards=len(jobs),native_rescue_identity_logits_max_abs=native_control_err,attention_source_reconstruction_relative=reconstruction,random_norm_relative=normerr,finite=all(np.isfinite(r['KL']) for r in records),no_training=True,test_evaluations=0)
 assert audits['hashes_unchanged'] and actual==len(jobs) and native_control_err<1e-4 and reconstruction<1e-5 and normerr<1e-5 and audits['finite']
 (ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,competence=panel['competence'],records=records,summary=summary,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
