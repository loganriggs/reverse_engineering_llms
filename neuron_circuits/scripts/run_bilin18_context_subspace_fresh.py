"""H559: fit context PCA on lexical discovery blocks; test response and KL fidelity."""
import json,hashlib,time
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure,continuation_id
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_context_subspace_fresh'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_context_subspace_transfer.json';tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);rows=[];excluded=[]
 names=[('William','Rachel'),('Patrick','Diana'),('Samuel','Julia'),('Dennis','Linda')];colors=[('amber','beige'),('coral','ivory'),('crimson','scarlet'),('teal','navy')]
 for item,((a,b),(v,w)) in enumerate(zip(names,colors)):
  for fam,prompts,answers in [('color_prose',[f'The code was {x}. The code was {x}. The code was' for x in (v,w)],[v,w]),('color_record',[f'record {x} | record {x} | record' for x in (v,w)],[v,w]),('name_prose',[f'The code was {x}. The code was {x}. The code was' for x in (a,b)],[a,b]),('name_record',[f'record {x} | record {x} | record' for x in (a,b)],[a,b]),('binding',[f'{a} and {b} entered the room. {g} passed the note to' for g in (a,b)],[b,a])]:
   ids=[tok.encode(t,add_special_tokens=False) for t in prompts];gs=[[[continuation_id(tok,t,' '+ans)] for ans in answers] for t in prompts]
   if len(ids[0])!=len(ids[1]) or any(g[0] is None for gg in gs for g in gg):excluded.append(f'{fam}-{item}');continue
   dif=[i for i,(x,y) in enumerate(zip(*ids)) if x!=y]
   for v in (0,1):rows.append(dict(id=f'{fam}-fresh{item}-{v}',pair=f'{fam}-fresh{item}',family=fam,variant=v,ids=ids[v],prompt=prompts[v],groups=gs[v],source_position=dif[-1]))
 test=rows;train=[];inputs={};native={};
 cfg=dict(experiment='H559',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),train_items=[0,1,2,3],test_items=[4,5,6,7],family_ranks=[0,1,2,4,7],global_ranks=[4,8,16,32],scales=[.0625,1.],design='New lexical combinations in the unchanged five templates. Freeze H558 family/global PCA bases and all ranks before new model evaluation; no fitting. Retain exact recipient normalized displacement and quadratic term.',metrics=['relative internal response error','answer margin MAE','KL(exact perturbed distribution || approximate perturbed distribution)','KL ratio to zero MLP response baseline'],scope='New constructed prompts; lexical words may have appeared in earlier unrelated experiments or model training. Only local context modulation compressed, not full MLP or downstream circuit. Token-invalid pairs excluded before inference; no competence filtering.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 m=model.blocks[8].mlp;block=model.blocks[8];D=model.config.n_embd//model.config.n_head;h=3
 (ART/f'{STEM}_manifest.json').write_text(json.dumps(rows,indent=2))
 bp=ART/'bilin18_context_subspace_transfer_bases.pt';bases=torch.load(bp,weights_only=False);bh=hashlib.sha256(bp.read_bytes()).hexdigest()
 print('FROZEN BASES',bh,'PROMPTS',len(rows),'EXCLUDED',excluded,flush=True)
 for r in rows:
  cap={}
  def pre(mod,args):cap['x']=args[0][:,-1].clone();cap['x0']=args[2][:,-1].clone()
  def ah(mod,args,out):cap['y']=out[0][:,-1].clone()
  handles=[block.register_forward_pre_hook(pre),block.attn.register_forward_hook(ah)]
  try:zz,cc=run(model,torch.tensor([r['ids']],device='cuda'),capture=True)
  finally:
   for handle in handles:handle.remove()
  native[r['id']]=measure(zz[0],r['groups']);x=(block.lambdas[0]*cap['x']+block.lambdas[1]*cap['x0']+cap['y'])[0];T=len(r['ids']);p=r['source_position'];raw={n:cc[(8,n)].cuda().view(1,T,model.config.n_head,D) for n in ('c_q','c_q2','c_k','c_k2')};cos,sin=block.attn.rotary(raw['c_q']);vec={}
  for n,v in raw.items():
   if n.startswith('c_k'):v=v[:,p:p+1].expand(-1,T,-1,-1)
   vec[n]=model.tt.apply_rotary_emb(F.rms_norm(v,(D,)),cos,sin)[0,:,h]
  def coef(pos):return (vec['c_q'][-1]@vec['c_k'][pos]/D)*(vec['c_q2'][-1]@vec['c_k2'][pos]/D)*block.attn.lamb
  delta=(coef(T-1)-coef(p))*F.linear(cc['bus'][0,p,h].cuda(),block.attn.c_proj.weight[:,h*D:(h+1)*D]);inputs[r['id']]=dict(x=x.cpu(),delta=delta.cpu())
 z={k:F.rms_norm(v['x'].cuda(),(v['x'].numel(),)) for k,v in inputs.items()}
 result=dict(config=cfg,excluded_pairs=excluded,native=[dict(id=r['id'],family=r['family'],variant=r['variant'],**native[r['id']]) for r in rows],bases_sha256=bh,geometry=[],interventions=[],status='running');jobs=defaultdict(list);exactlog={};nativekl={};scored={}
 def kl(a,b):
  la=a.double().log_softmax(-1);lb=b.double().log_softmax(-1);return float((la.exp()*(la-lb)).sum())
 for r in test:
  x=inputs[r['id']]['x'].cuda();base=z[r['id']];y0=m(base);contexts={}
  for kind,ranks in [('family',cfg['family_ranks']),('global',cfg['global_ranks'])]:
   b=bases[r['family'] if kind=='family' else 'global'];center=base-b['mean']
   for rank in ranks:V=b['V'][:rank];contexts[f'{kind}_{rank}']=b['mean']+(center@V.T)@V
  ids=torch.tensor([r['ids']],device='cuda');nat,_=run(model,ids)
  for scale in cfg['scales']:
   delta=scale*inputs[r['id']]['delta'].cuda();u=F.rms_norm(x+delta,(x.numel(),));e=u-base;actual=m(u)-y0;quad=m.Down(m.Left(e)*m.Right(e));responses=dict(exact=actual,zero=torch.zeros_like(actual))
   for mode,c in contexts.items():responses[mode]=m.Down(m.Left(c)*m.Right(e)+m.Left(e)*m.Right(c))+quad
   logits,_=run(model,ids,patch=(8,delta[None]));exactlog[r['id'],scale]=logits[0].cpu();nativekl[r['id'],scale]=kl(logits[0],nat[0]);scored[r['id'],scale]=measure(logits[0],r['groups'])['margin']
   for mode,response in responses.items():
    result['geometry'].append(dict(id=r['id'],family=r['family'],scale=scale,mode=mode,relative_response_error=float((response-actual).norm()/actual.norm())));jobs[len(r['ids'])].append((r,scale,mode,delta.cpu(),(y0+response).cpu()))
 for T,jj in jobs.items():
  for begin in range(0,len(jj),24):
   bb=jj[begin:begin+24]
   def hook(mod,args,out):out=out.clone();out[:,-1]=torch.stack([y for r,s,mode,d,y in bb]).to(out);return out
   handle=m.register_forward_hook(hook)
   try:zz,_=run(model,torch.tensor([r['ids'] for r,s,mode,d,y in bb],device='cuda'),patch=(8,torch.stack([d for r,s,mode,d,y in bb])))
   finally:handle.remove()
   for ix,(r,scale,mode,d,y) in enumerate(bb):result['interventions'].append(dict(id=r['id'],family=r['family'],scale=scale,mode=mode,margin_error=measure(zz[ix],r['groups'])['margin']-scored[r['id'],scale],kl=kl(exactlog[r['id'],scale].cuda(),zz[ix]),exact_to_native_kl=nativekl[r['id'],scale]))
 result['summary']={}
 for fam in sorted({r['family'] for r in test}):
  result['summary'][fam]={}
  for scale in cfg['scales']:
   result['summary'][fam][str(scale)]={};cc=[c for c in result['interventions'] if c['family']==fam and c['scale']==scale];zero=np.mean([c['kl'] for c in cc if c['mode']=='zero'])
   for mode in responses:
    vv=[c for c in cc if c['mode']==mode];gg=[g for g in result['geometry'] if g['family']==fam and g['scale']==scale and g['mode']==mode];mean_kl=float(np.mean([c['kl'] for c in vv]));result['summary'][fam][str(scale)][mode]=dict(relative_response_error=float(np.mean([g['relative_response_error'] for g in gg])),margin_mae=float(np.mean([abs(c['margin_error']) for c in vv])),mean_kl=mean_kl,kl_ratio_to_zero=float(mean_kl/zero),mean_exact_to_native_kl=float(np.mean([c['exact_to_native_kl'] for c in vv])))
 result['audits']=dict(exact_margin_max_error=max(abs(c['margin_error']) for c in result['interventions'] if c['mode']=='exact'),exact_kl_max=max(c['kl'] for c in result['interventions'] if c['mode']=='exact'),min_kl=min(c['kl'] for c in result['interventions']),finite=all(np.isfinite(c['kl']) for c in result['interventions']),bases_unchanged=bh==hashlib.sha256(bp.read_bytes()).hexdigest(),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),no_refitting=True);result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['audits']),flush=True)
if __name__=='__main__':main()
