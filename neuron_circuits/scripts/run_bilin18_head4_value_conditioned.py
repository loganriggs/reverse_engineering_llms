"""H661: single exact attention-layer repairs and leave-one-predicted controls."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
STEM='bilin18_head4_value_conditioned'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_operation_paraphrase_panel_manifest.json','captures':ART/'bilin18_operation_paraphrase_panel_captures.pt','means':ART/'bilin18_boundary_attention_wording_locked.pt','exact':ART/'bilin18_finite_factorial_captures.pt','previous':ART/'bilin18_layer14_head_repairs_captures.pt','runner':Path(__file__),'role_reference':ART/'bilin18_head4_token_roles_captures.pt'};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['captures'],weights_only=False);means=torch.load(paths['means'],weights_only=False)['new'];exact=torch.load(paths['exact'],weights_only=False);source,target=exact['source'],exact['target'];masks=['baseline','mean_coefficient','fixed_product','exact_coefficient','exact_value','exact_input']
 cfg=dict(experiment='H665',hashes=hashes,design='Factor aligned input writes of layer14 head4 into signed coefficients and projected effective values. Fit mean coefficient change and mean receiver value per input-token ordinal from first4 discovery items only. Apply mean coefficient change to heldout receiver values; compare unconditional mean baseline, product of discovery means, exact-coefficient/receiver-value, receiver-coefficient/exact-value and exact-input controls. Keep all other mean predictions fixed; predicted/exact boundary.',limitations='Known task/form, frozen templates, retrospective panel. Oracle controls donor-informed; mean-coefficient prediction uses receiver factors only. Multi-token inputs matched by complete token-ID sequence, fixed count per family; no output fitting.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 buckets=defaultdict(list)
 for i,s in enumerate(source):buckets[len(rows[s]['ids'])].append(i)
 basea=torch.empty(96,8,1152);attnevals=0
 for l in range(10,18):
  block=model.blocks[l]
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];ss=[source[i] for i in bb];x=torch.stack([cap['caches'][l][i] for i in ss]).cuda();x0=torch.stack([cap['x0'][i] for i in ss]).cuda();bus=torch.stack([cap['bus'][i] for i in ss]).cuda();a,_=block.attn(F.rms_norm(block.lambdas[0]*x+block.lambdas[1]*x0,(1152,)),bus);basea[bb,l-10]=a[:,-1].cpu();attnevals+=len(bb)
 lookup={(r['family'],r['control'],r['item'],r['form']):i for i,r in enumerate(rows)};roles=['instruction','input_label','input','suffix'];tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);groups={};tokenmeta={};hb=defaultdict(list);head_replay=[];prior_heads=torch.load(paths['previous'],weights_only=False)['projected_heads']
 for i,r in enumerate(rows):
  if r['control'] not in ['task','copy']:continue
  encoded=tok(r['prompt'],add_special_tokens=False,return_offsets_mapping=True);assert encoded['input_ids']==r['ids'];start=r['prompt'].index('\nInput: ');content=start+len('\nInput: ');end=content+len(r['input']);assert r['prompt'][content:end]==r['input'];labels=[]
  def role_at(pos):return 0 if pos<start else 1 if pos<content else 2 if pos<end else 3
  for lo,hi in encoded['offset_mapping']:
   nonwhite=[p for p in range(lo,hi) if not r['prompt'][p].isspace()];rr={role_at(p) for p in nonwhite};assert len(rr)<=1,(r['id'],lo,hi,rr);labels.append(next(iter(rr)) if rr else role_at(lo))
  tokenmeta[i]=dict(id=r['id'],labels=labels,tokens=tok.convert_ids_to_tokens(r['ids']),offsets=encoded['offset_mapping']);hb[(r['item']<4,len(r['ids']))].append(i)
 block=model.blocks[14];att=block.attn;assert att.squared_attn and att.c_proj.bias is None;W=att.c_proj.weight[:,4*128:5*128].double();position_writes={};coefficients={};projected_values={}
 for ii in hb.values():
  for st in range(0,len(ii),16):
   bb=ii[st:st+16];x=torch.stack([cap['caches'][14][i] for i in bb]).cuda();x0=torch.stack([cap['x0'][i] for i in bb]).cuda();bus=torch.stack([cap['bus'][i] for i in bb]).cuda();y=F.rms_norm(block.lambdas[0]*x+block.lambdas[1]*x0,(1152,));B,T,_=y.shape
   q,k,q2,k2=[mod(y).view(B,T,9,128) for mod in [att.c_q,att.c_k,att.c_q2,att.c_k2]];cos,sin=att.rotary(q);q,k,q2,k2=[model.tt.apply_rotary_emb(F.rms_norm(v,(128,)),cos,sin) for v in [q,k,q2,k2]];c1=(q[:,-1,4,None,:]*k[:,:,4]).sum(-1)/128;c2=(q2[:,-1,4,None,:]*k2[:,:,4]).sum(-1)/128;coeff=c1*c2;values=(1-att.lamb)*att.c_v(y).view(B,T,9,128)+att.lamb*bus.view(B,T,9,128);writes=F.linear(values[:,:,4].double()*coeff.double().unsqueeze(-1),W);attnevals+=len(bb)
   for n,i in enumerate(bb):
    w=writes[n].cpu();projected_values[i]=F.linear(values[n,:,4].double(),W).cpu();position_writes[i]=w;coefficients[i]=coeff[n].cpu();truth=prior_heads[i][4];head_replay.append(float((w.sum(0)-truth).norm()/truth.norm()));groups[i]=torch.stack([w[torch.tensor(tokenmeta[i]['labels'])==role].sum(0) for role in range(4)])
 assert max(head_replay)<1e-4
 rolemeans={}
 for family in sorted({r['family'] for r in rows}):
  for form in range(4):rolemeans[family,form]=torch.stack([groups[lookup[family,'task',i,form]]-groups[lookup[family,'copy',i,form]] for i in range(4)]).mean(0)
 roleres=torch.stack([groups[t]-groups[s]-rolemeans[rows[s]['family'],rows[s]['form']] for s,t in zip(source,target)]);priorres=torch.load(paths['previous'],weights_only=False)['head_residuals'][:,4];residual_replay=float(((roleres.sum(1)-priorres).norm(dim=-1)/priorres.norm(dim=-1)).max());assert residual_replay<1e-3
 aligned={};models={};factor_audit=[]
 for family in sorted({r['family'] for r in rows}):
  for form in range(4):
   for item in range(8):
    si=lookup[family,'copy',item,form];ti=lookup[family,'task',item,form];sp=[k for k,v in enumerate(tokenmeta[si]['labels']) if v==2];tp=[k for k,v in enumerate(tokenmeta[ti]['labels']) if v==2];assert [rows[si]['ids'][k] for k in sp]==[rows[ti]['ids'][k] for k in tp];aligned[family,form,item]=(si,ti,sp,tp)
   ds=[];vs=[]
   for item in range(4):
    si,ti,sp,tp=aligned[family,form,item];ds.append(coefficients[ti][tp].double()-coefficients[si][sp].double());vs.append(projected_values[si][sp])
   assert len({len(v) for v in ds})==1;models[family,form]=dict(coefficient_delta=torch.stack(ds).mean(0),receiver_value=torch.stack(vs).mean(0))
 adjustments={m:[] for m in masks};geometry=[]
 for i,sr in enumerate(source):
  row=rows[sr];family,form,item=row['family'],row['form'],row['item'];si,ti,sp,tp=aligned[family,form,item];cr=coefficients[si][sp].double();ct=coefficients[ti][tp].double();vr=projected_values[si][sp];vt=projected_values[ti][tp];fitted=models[family,form];dc=fitted['coefficient_delta'];assert len(dc)==len(cr)
  original_mean=rolemeans[family,form][2];truth=(ct[:,None]*vt-cr[:,None]*vr).sum(0);factor_audit.append(dict(id=row['id'],family=family,value_relative_change=float((vt-vr).norm()/vr.norm()),input_tokens=len(sp)))
  predictions={'baseline':original_mean,'mean_coefficient':(dc[:,None]*vr).sum(0),'fixed_product':(dc[:,None]*fitted['receiver_value']).sum(0),'exact_coefficient':((ct-cr)[:,None]*vr).sum(0),'exact_value':(cr[:,None]*(vt-vr)).sum(0),'exact_input':truth}
  for mode,v in predictions.items():adjustments[mode].append(v-original_mean);geometry.append(dict(id=row['id'],family=family,mode=mode,squared_error=float((v-truth).square().sum()),target_squared=float(truth.square().sum())))
 adjustments={k:torch.stack(v) for k,v in adjustments.items()}
 predb=cap['native_captures'][9][source].double()+torch.stack([means[rows[s]['family'],rows[s]['form']]['boundary'] for s in source]);preda=basea.double()+torch.stack([means[rows[s]['family'],rows[s]['form']]['attention'] for s in source]);donorb=cap['native_captures'][9][target];donora=torch.stack([exact['donor_attention'][l] for l in range(10,18)],1);forwards=0
 def evaluate(boundary,mask):
  nonlocal forwards
  logits=torch.empty(96,model.lm_head.weight.shape[0])
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];hooks=[]
    def bpatch(mod,args,out):
     y,bus=out;y=y.clone();y[:,-1]=(donorb if boundary=='exact' else predb)[bb].to(y);return y,bus
    hooks.append(model.blocks[9].register_forward_hook(bpatch))
    for l in range(10,18):
     def apatch(mod,args,out,l=l):
      y,bus=out;y=y.clone();v=preda[bb,l-10]
      if l==14:v=v+adjustments[mask][bb]
      y[:,-1]=v.to(y);return y,bus
     hooks.append(model.blocks[l].attn.register_forward_hook(apatch))
    try:z,_=run(model,torch.tensor([rows[source[i]]['ids'] for i in bb],device='cuda'));logits[bb]=z.cpu()
    finally:
     for h in hooks:h.remove()
    forwards+=len(bb)
  return logits
 tasklog=exact['donor_logits'].double().log_softmax(-1);p=tasklog.exp();records=[];outputs={}
 for b in ['predicted','exact']:
  for mask in masks:
   z=evaluate(b,mask);outputs[b,mask]=z;lp=z.double().log_softmax(-1);kl=(p*(tasklog-lp)).sum(-1)
   for i,s in enumerate(source):
    r=rows[s];t=rows[target[i]];ids=t['candidate_ids'];aid=t['answer_id'];records.append(dict(id=r['id'],family=r['family'],form=r['form'],item=r['item'],boundary=b,mask=mask,task_kl=float(kl[i]),target_ce=float(-lp[i,aid]),candidate_correct=ids[int(z[i,ids].argmax())]==aid,top1_correct=int(z[i].argmax())==aid))
  print('COMPLETE',b,flush=True)
 prior=torch.load(paths['role_reference'],weights_only=False)['logits'];replays={str((b,m)):float((outputs[b,m]-prior[b,0 if m=='baseline' else 4]).abs().max()) for b in ['predicted','exact'] for m in ['baseline','exact_input']};assert max(replays.values())<1e-3;audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},whole_model_forwards=forwards,cached_attention_evaluations=attnevals,head_sum_max_relative_error=max(head_replay),residual_sum_max_relative_error=residual_replay,previous_baseline_replay_max_abs=replays,finite=all(np.isfinite(r['task_kl']) for r in records));torch.save(dict(logits=outputs,role_residuals=roleres,role_means=rolemeans,token_metadata=tokenmeta,position_writes=position_writes,coefficients=coefficients,source=source,target=target,factor_models=models,projected_values=projected_values),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,geometry=geometry,factor_audit=factor_audit,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()
