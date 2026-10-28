"""H608 layer trace and final-position removal of prefix/case interaction writes."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_prefix_interaction_trace'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_prefix_case_factorial_manifest.json';rows=json.loads(src.read_text());ex={r['id']:r for r in rows};index={(r['wording'],r['item'],r['variant'],r['prefix'],r['lower']):r['id'] for r in rows};ops=[(kind,l) for l in range(11) for kind in ['attn','mlp']]+[('attn',11)];cfg=dict(experiment='H608',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design='Trace four-condition interaction at final-position attention/MLP writes and residuals. On combined prefix+lowercase prompt subtract each native write interaction separately before MLP11 input: attention/MLP0-10 and attention11. Compare one seeded norm-matched random subtraction per site and identity run. All23 sites predeclared, no ranking-based filtering.',scope='Removing interaction at a write is not removing its local generation: inherited interaction can appear in the write. Only final position intervened, other positions remain live. Target is native normalized MLP11 input interaction, not H607 injected-signal response. Full output KL is relative to unmodified combined prompt.',seed=608)
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 features={};native={};batches=defaultdict(list)
 for k,e in ex.items():batches[len(e['ids'])].append(k)
 for kk in batches.values():
  for st in range(0,len(kk),16):
   bb=kk[st:st+16];cap={};hs=[]
   for l,b in enumerate(model.blocks):
    def pre(mod,args,l=l):cap[f'pre{l}']=args[0][:,-1].cpu();cap['x0']=args[2][:,-1].cpu()
    def post(mod,args,out,l=l):cap[f'post{l}']=out[0][:,-1].cpu()
    def ah(mod,args,out,l=l):cap[f'attn{l}']=out[0][:,-1].cpu()
    def mh(mod,args,out,l=l):cap[f'mlp{l}']=out[:,-1].cpu();cap[f'input{l}']=args[0][:,-1].cpu()
    hs.extend([b.register_forward_pre_hook(pre),b.register_forward_hook(post),b.attn.register_forward_hook(ah),b.mlp.register_forward_hook(mh)])
   try:z,_=run(model,torch.tensor([ex[k]['ids'] for k in bb],device='cuda'))
   finally:
    for h in hs:h.remove()
   for i,k in enumerate(bb):features[k]={n:v[i].clone() for n,v in cap.items()};native[k]=z[i].cpu()
 interactions={};trace=[];account=[];combined=[]
 for w in [0,1]:
  for item in range(8):
   for v in [0,1]:
    ids=[index[w,item,v,p,c] for p,c in [(0,0),(0,1),(1,0),(1,1)]];k=ids[3];combined.append(k);a,b,c,d=[features[ii] for ii in ids];ints={n:d[n]-c[n]-b[n]+a[n] for n in a};interactions[k]=ints;total=d['input11']-a['input11']
    for l,block in enumerate(model.blocks):
     predicted=float(block.lambdas[0])*ints[f'pre{l}']+float(block.lambdas[1])*ints['x0']+ints[f'attn{l}']+ints[f'mlp{l}'];actual=ints[f'post{l}'];account.append(float((actual-predicted).norm()/actual.norm().clamp_min(1e-10)))
     trace.append(dict(id=k,wording=w,layer=l,input_interaction_norm=float(ints[f'input{l}'].norm()),pre_residual_interaction_norm=float(ints[f'pre{l}'].norm()),post_residual_interaction_norm=float(actual.norm()),attention_interaction_norm=float(ints[f'attn{l}'].norm()),mlp_interaction_norm=float(ints[f'mlp{l}'].norm()),target_norm=float(ints['input11'].norm()),total_context_norm=float(total.norm())))
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);token_audit=[]
 for w in [0,1]:
  ids=[index[w,0,0,p,c] for p,c in [(0,0),(0,1),(1,0),(1,1)]];seq=[ex[k]['ids'] for k in ids];offset=len(seq[2])-len(seq[0]);assert len(seq[0])==len(seq[1]) and len(seq[2])==len(seq[3]);token_audit.append(dict(wording=w,prefix_token_offset=offset,first_body_token_ids=[seq[0][0],seq[1][0],seq[2][offset],seq[3][offset]],first_body_tokens=tok.convert_ids_to_tokens([seq[0][0],seq[1][0],seq[2][offset],seq[3][offset]]),body_tokens_changed_by_prefix=[i for i,(a,b) in enumerate(zip(seq[0],seq[2][offset:])) if a!=b],body_tokens_changed_by_case=[i for i,(a,b) in enumerate(zip(seq[0],seq[1])) if a!=b]))
 print('TOKENS',token_audit,flush=True);result=dict(config=cfg,trace=trace,token_audit=token_audit,interventions=[],summary={},status='running');gen=torch.Generator().manual_seed(608);random={op:torch.randn(1152,generator=gen) for op in ops};random={op:v/v.norm() for op,v in random.items()};jobs=defaultdict(list)
 for k in combined:jobs[len(ex[k]['ids'])].append(k)
 replay=[]
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 save()
 for kind,l in [('self',-1)]+ops:
  for mode in (['self'] if kind=='self' else ['interaction','random']):
   for kk in jobs.values():
    for st in range(0,len(kk),16):
     bb=kk[st:st+16];cap={};hs=[]
     def target_hook(m,args):cap['target']=args[0][:,-1].cpu()
     hs.append(model.blocks[11].mlp.register_forward_pre_hook(target_hook))
     if kind!='self':
      iv=torch.stack([interactions[k][f'{kind}{l}'] for k in bb]);delta=iv if mode=='interaction' else iv.norm(dim=-1,keepdim=True)*random[kind,l]
      if kind=='attn':
       def edit(m,args,out):y,bus=out;y=y.clone();y[:,-1]-=delta.to(y);return y,bus
      else:
       def edit(m,args,out):y=out.clone();y[:,-1]-=delta.to(y);return y
      hs.append(getattr(model.blocks[l],kind).register_forward_hook(edit))
     try:z,_=run(model,torch.tensor([ex[k]['ids'] for k in bb],device='cuda'))
     finally:
      for h in hs:h.remove()
     for i,k in enumerate(bb):
      e=ex[k];orig=features[k]['input11'];I=interactions[k]['input11'];newI=I+cap['target'][i]-orig;shift=cap['target'][i]-orig;zz=z[i].cpu();rp=native[k].double().log_softmax(0);lp=zz.double().log_softmax(0);target=e['groups'][e['variant']][0]
      if kind=='self':replay.append(float((zz-native[k]).abs().max()))
      result['interventions'].append(dict(id=k,wording=e['wording'],kind=kind,layer=l,mode=mode,target_interaction_ratio=float(newI.norm()/I.norm()),interaction_alignment_removed=float(-(shift@I)/I.square().sum()),target_shift_ratio=float(shift.norm()/I.norm()),reference_kl=float((rp.exp()*(rp-lp)).sum()),margin_delta=measure(zz,e['groups'])['margin']-measure(native[k],e['groups'])['margin'],target_ce_delta=float(rp[target]-lp[target])))
   save()
  print('COMPLETE',kind,l,flush=True)
 for w in [0,1]:
  for kind,l in ops:
   for mode in ['interaction','random']:
    vv=[c for c in result['interventions'] if c['wording']==w and c['kind']==kind and c['layer']==l and c['mode']==mode];result['summary'][f'{w}_{kind}{l}_{mode}']={n:float(np.mean([c[n] for c in vv])) for n in ['target_interaction_ratio','interaction_alignment_removed','target_shift_ratio','reference_kl','target_ce_delta']}
 result['audits']=dict(runner_unchanged=cfg['runner_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),native_prompts=len(ex),interventions=len(result['interventions']),identity_replay_max_logit_error=max(replay),residual_accounting_max_relative_error=max(account),x0_interaction_max_norm=max(float(v['x0'].norm()) for v in interactions.values()),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['target_interaction_ratio','reference_kl','target_ce_delta']));torch.save(dict(features=features,native=native,interactions=interactions,random=random),ART/f'{STEM}_captures.pt');result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
