"""H793 stagewise factorial interaction trace; no additional intervention selection."""
import json,time,math
from pathlib import Path
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
S='bilin18_source_interaction_trace'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'captures':ART/'bilin18_source_donor_specificity_captures.pt','reference':ART/'bilin18_source_identity_composition.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};c=torch.load(paths['captures'],weights_only=False);old=json.loads(paths['reference'].read_text());idx={(r['id'],r['a'],r['b']):r for r in old['records']};cfg={'experiment':'H793','hashes':hashes,'layers':list(range(1,18)),'positions':['first_source','second_source','query'],'stages':['block_scaled_input','attn_normalized_input','attn_output','pre_mlp_residual','mlp_normalized_input','mlp_output','block_output'],'design':'Replay full H792 4x4 source factorial for each item in one batch. Capture each stage at both source positions and final token. Quantify I(a,b)=z_ab-z_a0-z_0b+z_00, compare with actual joint change and native magnitude. Trace only, no interaction-removal intervention.','predictions':['First source cannot depend on the later source under causal masking.','Immediately after L1 replacements interactions vanish.','At layer2 query attention contributions from two altered sources should add, since query input is unchanged and attention has no softmax coupling. Normalization and/or subsequent MLP can create query interaction.'],'limitations':'Stagewise magnitude is not causal mediation or importance. Different stages have different scales; ratios need actual joint-change denominators. Reused panel; source-position causality is an architectural audit.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 records=[];replay=[];identity=[];initial=[];causal=[];queryadd=[];snapshots={};forwards=0
 for rid in sorted({k[0] for k in c['examples']}):
  e=c['examples'][rid,0];positions=e['positions'];jobs=[(a,b) for a in range(4) for b in range(4)];saved={};raw={};handles=[]
  def inject(m,args,out):
   y=out.clone()
   for i,(a,b) in enumerate(jobs):y[i,positions[0]]=c['writes'][rid,a][0].to(y);y[i,positions[1]]=c['writes'][rid,b][1].to(y)
   return y
  handles.append(model.blocks[1].mlp.register_forward_hook(inject))
  for l in range(1,18):
   def preblock(m,args,l=l):raw[l]=m.lambdas[0]*args[0]+m.lambdas[1]*args[2];saved[l,'block_scaled_input']=raw[l][:,positions].cpu()
   def preattn(m,args,l=l):saved[l,'attn_normalized_input']=args[0][:,positions].cpu()
   def postattn(m,args,out,l=l):saved[l,'attn_output']=out[0][:,positions].cpu();saved[l,'pre_mlp_residual']=(raw[l]+out[0])[:,positions].cpu()
   def premlp(m,args,l=l):saved[l,'mlp_normalized_input']=args[0][:,positions].cpu()
   def postmlp(m,args,out,l=l):saved[l,'mlp_output']=out[:,positions].cpu()
   def postblock(m,args,out,l=l):saved[l,'block_output']=out[0][:,positions].cpu()
   block=model.blocks[l];handles.extend([block.register_forward_pre_hook(preblock),block.attn.register_forward_pre_hook(preattn),block.attn.register_forward_hook(postattn),block.mlp.register_forward_pre_hook(premlp),block.mlp.register_forward_hook(postmlp),block.register_forward_hook(postblock)])
  try:logits,_=run(model,torch.tensor([e['ids']]*len(jobs),device='cuda'))
  finally:
   for h in handles:h.remove()
  forwards+=len(jobs);ids=e['candidate_ids']
  for i,(a,b) in enumerate(jobs):
   delta=(logits[i].cpu()-c['native'][rid,0])[ids];replay.append(float((delta-torch.tensor(idx[rid,a,b]['candidate_logit_changes'])).abs().max()))
   if a==0 and b==0:identity.append(float((logits[i].cpu()-c['native'][rid,0]).abs().max()))
  snapshots[rid]={k:v for k,v in saved.items() if k[0] in [1,2,3]}
  for (l,stage),tensor in saved.items():
   z=tensor.double().reshape(4,4,3,-1);base=z[0,0]
   for kind in ['coherent','mixed']:
    pp=[(a,b) for a in range(1,4) for b in range(1,4) if (a==b)==(kind=='coherent')];inter=torch.stack([z[a,b]-z[a,0]-z[0,b]+base for a,b in pp]);actual=torch.stack([z[a,b]-base for a,b in pp]);signal=torch.stack([z[a,0]-base for a,b in pp]);scale=torch.stack([z[a,b] for a,b in pp])
    for role,name in enumerate(cfg['positions']):
     v=inter[:,role];d=actual[:,role];records.append({'id':rid,'item':e['item'],'format':e['format'],'layer':l,'stage':stage,'kind':kind,'position':name,'cases':len(pp),'interaction_squared_norm':float(v.square().sum()),'actual_joint_change_squared_norm':float(d.square().sum()),'native_squared_norm':float(base[role].square().sum())*len(pp),'state_squared_norm':float(scale[:,role].square().sum()),'interaction_max_abs':float(v.abs().max())})
    if l==1 and stage=='block_output':initial.append(float(inter.abs().max()))
    causal.append(float((z[:,:,0]-z[:,0:1,0]).abs().max()))
    if l==2 and stage=='attn_output':queryadd.append({'numerator':float(inter[:,2].square().sum()),'denominator':float(actual[:,2].square().sum()),'max_abs':float(inter[:,2].abs().max())})
  print('traced',rid,flush=True)
 queryrelative=(sum(r['numerator'] for r in queryadd)/sum(r['denominator'] for r in queryadd))**.5;audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'H792_candidate_delta_replay_max_abs':max(replay),'identity_logits_max_abs':max(identity),'L1_post_patch_factorial_max_abs':max(initial),'first_source_future_independence_max_abs':max(causal),'L2_query_attention_interaction_relative_RMS':queryrelative,'L2_query_attention_interaction_max_abs':max(r['max_abs'] for r in queryadd),'finite':all(math.isfinite(r['interaction_squared_norm']) for r in records),'whole_model_forwards':forwards,'test_evaluations':0};print(audits,flush=True);assert audits['hashes_unchanged'] and audits['finite'] and max(replay+identity)<1e-4 and max(initial)<1e-5 and max(causal)<1e-5 and queryrelative<1e-4
 torch.save(snapshots,ART/f'{S}_early_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
if __name__=='__main__':main()
