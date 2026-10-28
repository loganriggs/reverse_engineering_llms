"""H666: frozen coefficient/value rule, prospective new item panel."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
STEM='bilin18_coefficient_task_transfer'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_value_rule_fresh_panel_manifest.json','captures':ART/'bilin18_value_rule_fresh_panel_captures.pt','frozen_factors':ART/'bilin18_head4_value_conditioned_captures.pt','frozen_means':ART/'bilin18_boundary_attention_wording_locked.pt','preregistration':ART/'bilin18_value_rule_fresh_preregistration.json','runner':Path(__file__),'prior':ART/'bilin18_value_rule_fresh_transfer_captures.pt'};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};freeze=json.loads(paths['preregistration'].read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in freeze['hashes'].items());rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['captures'],weights_only=False);frozen=torch.load(paths['frozen_factors'],weights_only=False);models,rolemeans=frozen['factor_models'],frozen['role_means'];del frozen;means=torch.load(paths['frozen_means'],weights_only=False)['new'];lookup={(r['family'],r['control'],r['item'],r['form']):i for i,r in enumerate(rows)};source=[i for i,r in enumerate(rows) if r['control']=='copy'];target=[lookup[rows[s]['family'],'task',rows[s]['item'],rows[s]['form']] for s in source];assert len(source)==96
 cfg=dict(experiment='H667',hashes=hashes,design='Frozen H665 coefficients: compare own-task vs leave-target-family-out pooled coefficients and individual source-family scalars from plural/antonym/capital/successor, matched by form. Apply to fresh receiver input values with other components fixed. Four single-token families primary; addition/agreement apply scalar to every input position as explicit broadcast stress test. Baseline/zero-change controls and exact boundary controls. No fitting.',limitations='H666 panel now inspected, so retrospective family-transfer diagnostic. Other components retain known target-family identity. Source means based on only4 discovery items; no semantic-unit claim.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 indices=source+target;buckets=defaultdict(list)
 for i in indices:buckets[len(rows[i]['ids'])].append(i)
 attnevals=0;nativea={l:{} for l in range(10,18)}
 for l in range(10,18):
  block=model.blocks[l]
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];x=torch.stack([cap['caches'][l][i] for i in bb]).cuda();x0=torch.stack([cap['x0'][i] for i in bb]).cuda();bus=torch.stack([cap['bus'][i] for i in bb]).cuda();a,_=block.attn(F.rms_norm(block.lambdas[0]*x+block.lambdas[1]*x0,(1152,)),bus);attnevals+=len(bb)
    for k,i in enumerate(bb):nativea[l][i]=a[k,-1].cpu()
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);positions={};ivals={};iwrites={};headreplay=[];att=model.blocks[14].attn;W=att.c_proj.weight[:,512:640].double();assert att.squared_attn
 for i in indices:
  r=rows[i];e=tok(r['prompt'],add_special_tokens=False,return_offsets_mapping=True);assert e['input_ids']==r['ids'];lo=r['prompt'].index('\nInput: ')+len('\nInput: ');hi=lo+len(r['input']);pp=[]
  for k,(a,b) in enumerate(e['offset_mapping']):
   chars=[p for p in range(a,b) if not r['prompt'][p].isspace()]
   if any(lo<=p<hi for p in chars):assert all(lo<=p<hi for p in chars);pp.append(k)
  positions[i]=pp
 for ii in buckets.values():
  for st in range(0,len(ii),16):
   bb=ii[st:st+16];block=model.blocks[14];x=torch.stack([cap['caches'][14][i] for i in bb]).cuda();x0=torch.stack([cap['x0'][i] for i in bb]).cuda();bus=torch.stack([cap['bus'][i] for i in bb]).cuda();y=F.rms_norm(block.lambdas[0]*x+block.lambdas[1]*x0,(1152,));B,T,_=y.shape;q,k,q2,k2=[m(y).view(B,T,9,128) for m in [att.c_q,att.c_k,att.c_q2,att.c_k2]];cos,sin=att.rotary(q);q,k,q2,k2=[model.tt.apply_rotary_emb(F.rms_norm(v,(128,)),cos,sin) for v in [q,k,q2,k2]];coeff=((q[:,-1,4,None]*k[:,:,4]).sum(-1)/128)*((q2[:,-1,4,None]*k2[:,:,4]).sum(-1)/128);v=(1-att.lamb)*att.c_v(y).view(B,T,9,128)+att.lamb*bus.view(B,T,9,128);pv=F.linear(v[:,:,4].double(),W);writes=pv*coeff.double().unsqueeze(-1);saved=[]
   def capture(m,args):saved.append(args[0][:,-1,512:640].clone())
   h=att.c_proj.register_forward_pre_hook(capture)
   try:att(y,bus)
   finally:h.remove()
   truth=F.linear(saved[0].double(),W);headreplay.append(float(((writes.sum(1)-truth).norm(dim=-1)/truth.norm(dim=-1)).max()));attnevals+=len(bb)
   for k,i in enumerate(bb):ivals[i]=pv[k,positions[i]].cpu();iwrites[i]=writes[k,positions[i]].sum(0).cpu()
 assert max(headreplay)<1e-4
 basea=torch.stack([torch.stack([nativea[l][s] for l in range(10,18)]) for s in source]);donora=torch.stack([torch.stack([nativea[l][t] for l in range(10,18)]) for t in target]);preda=basea.double()+torch.stack([means[rows[s]['family'],rows[s]['form']]['attention'] for s in source]);predb=cap['native_captures'][9][source].double()+torch.stack([means[rows[s]['family'],rows[s]['form']]['boundary'] for s in source]);donorb=cap['native_captures'][9][target];single_families=['plural','antonym','capital','successor'];assert all(len(models[f,form]['coefficient_delta'])==1 for f in single_families for form in range(4));adjust={m:[] for m in ['baseline','mean_coefficient','pooled_other','zero']+['from_'+f for f in single_families]};geometry=[];coefficient_table={f:[float(models[f,form]['coefficient_delta'][0]) for form in range(4)] for f in single_families}
 for s,t in zip(source,target):
  r=rows[s];f=models[r['family'],r['form']];assert [rows[s]['ids'][k] for k in positions[s]]==[rows[t]['ids'][k] for k in positions[t]];assert len(positions[s])==len(f['coefficient_delta']);old=rolemeans[r['family'],r['form']][2];truth=iwrites[t]-iwrites[s];sources=[sf for sf in single_families if sf!=r['family']];pooled=torch.stack([models[sf,r['form']]['coefficient_delta'][0] for sf in sources]).mean();preds={'baseline':old,'mean_coefficient':(f['coefficient_delta'][:,None]*ivals[s]).sum(0),'pooled_other':pooled*ivals[s].sum(0),'zero':torch.zeros_like(old)}
  for sf in single_families:preds['from_'+sf]=models[sf,r['form']]['coefficient_delta'][0]*ivals[s].sum(0)
  for m,v in preds.items():adjust[m].append(v-old);geometry.append(dict(id=r['id'],family=r['family'],mode=m,error_squared=float((v-truth).square().sum()),target_squared=float(truth.square().sum())))
 adjust={m:torch.stack(v) for m,v in adjust.items()};buckets=defaultdict(list)
 for i,s in enumerate(source):buckets[len(rows[s]['ids'])].append(i)
 forwards=0
 def evaluate(bmode,mode):
  nonlocal forwards
  zall=torch.empty(96,model.lm_head.weight.shape[0])
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];hooks=[]
    if bmode!='native':
     def bp(m,args,out):
      y,b=out;y=y.clone();y[:,-1]=(donorb if bmode=='exact' else predb)[bb].to(y);return y,b
     hooks.append(model.blocks[9].register_forward_hook(bp))
    if mode!='native':
     for l in range(10,18):
      def ap(m,args,out,l=l):
       y,b=out;y=y.clone();v=donora[bb,l-10] if mode=='all_exact' else preda[bb,l-10]
       if l==14 and mode!='all_exact':v=v+adjust[mode][bb]
       y[:,-1]=v.to(y);return y,b
      hooks.append(model.blocks[l].attn.register_forward_hook(ap))
    try:z,_=run(model,torch.tensor([rows[source[i]]['ids'] for i in bb],device='cuda'));zall[bb]=z.cpu()
    finally:
     for h in hooks:h.remove()
    forwards+=len(bb)
  return zall
 tasklog=cap['native_logits'][target].double().log_softmax(-1);p=tasklog.exp();records=[];outputs={};conditions=[('native','native'),('exact','all_exact')]+[(b,m) for b in ['predicted','exact'] for m in adjust]
 for b,m in conditions:
  z=evaluate(b,m);outputs[b,m]=z;lp=z.double().log_softmax(-1);kl=(p*(tasklog-lp)).sum(-1)
  for i,s in enumerate(source):
   r=rows[s];t=rows[target[i]];cs=t['candidate_ids'];aid=t['answer_id'];records.append(dict(id=r['id'],family=r['family'],item=r['item'],form=r['form'],boundary=b,mode=m,task_kl=float(kl[i]),target_ce=float(-lp[i,aid]),candidate_correct=cs[int(z[i,cs].argmax())]==aid,top1_correct=int(z[i].argmax())==aid))
  print('COMPLETE',b,m,flush=True)
 nativeerror=float((outputs['native','native']-cap['native_logits'][source]).abs().max());exacterror=float((outputs['exact','all_exact']-cap['native_logits'][target]).abs().max());assert nativeerror<1e-3 and exacterror<1e-3;prior=torch.load(paths['prior'],weights_only=False)['logits'];prior_diff=max(float((outputs[b,m]-prior[b,m]).abs().max()) for b in ['predicted','exact'] for m in ['baseline','mean_coefficient']);assert prior_diff==0;audits=dict(previous_prediction_max_abs=prior_diff,hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},preregistered_hashes_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in freeze['hashes'].items()),whole_model_forwards=forwards,cached_attention_evaluations=attnevals,head_reconstruction_max_relative=max(headreplay),native_replay_max_abs=nativeerror,exact_recovery_max_abs=exacterror,no_refit=True,finite=all(np.isfinite(r['task_kl']) for r in records));torch.save(dict(logits=outputs,source=source,target=target),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,geometry=geometry,coefficient_table=coefficient_table,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()
