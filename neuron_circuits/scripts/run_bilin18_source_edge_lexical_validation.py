"""H796 frozen L2H5 source-edge validation on separate household-word panel."""
import json,time,math
from pathlib import Path
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
S='bilin18_source_edge_lexical_validation'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'captures':ART/'bilin18_native_scale_replication_captures.pt','source_writes':ART/'bilin18_copy_source_query_localization_captures.pt','reference':ART/'bilin18_copy_source_query_localization.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};c=torch.load(paths['captures'],weights_only=False);src=torch.load(paths['source_writes'],weights_only=False);ref=json.loads(paths['reference'].read_text());examples={k:e for k,e in c['examples'].items() if e['task']=='sequence_copy' and k[1] in [0,1]};native=c['native'];writes={k:v[1] for k,v in src['writes'].items() if k in examples};arms=['none','all_edges','full_module','head5'];cfg={'experiment':'H796','hashes':hashes,'arms':arms,'design':'Replay source factorial; capture actual L2 post-normalization/RoPE Q,K,Q2,K2 and mixedV. Reconstruct first-source→second-source signed edge per head, project factorial interaction through c_proj. Remove one head interaction, all edge interactions, or full-module interaction at second-source output.','prediction':'Frozen L2H5 should reproduce full-edge interaction-removal effects on a separate lexical panel; no head reselection. Measure all-edge reconstruction and per-example candidate/margin fidelity. Matching vs mismatching signed routing is descriptive, not proof of semantic equality.','limitations':'Household panel previously used for task/site analysis but not H795 head selection. Same two formats; vocabulary differs. Only two identities per item, so no nonreceiver mixed-donor factorial cell. Partial native MLP-output substitutions remain hybrid states.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 attn=model.blocks[2].attn;assert attn.squared_attn;original=attn.squared_attention;W=attn.c_proj.weight.cpu().double();states={};patches={};geometry=[];reconstruction=[];independence=[];capture_replay=[];forwards=0;refidx={(r['id'],r['site']):r for r in ref['records'] if r['receiver']==0 and r['donor']==1 and r['layer']==1};jobs=[(a,b) for a in range(2) for b in range(2)]
 for rid in sorted({k[0] for k in examples}):
  e=examples[rid,0];p1,p2,_=src['positions'][rid,0];saved={}
  def inject(m,args,out):
   y=out.clone()
   for i,(a,b) in enumerate(jobs):y[i,p1]=writes[rid,a][0].to(y);y[i,p2]=writes[rid,b][1].to(y)
   return y
  def intercept(q,k,v,q2,k2):
   pattern=(torch.einsum('bqhd,bkhd->bhqk',q,k)/128)*(torch.einsum('bqhd,bkhd->bhqk',q2,k2)/128);saved['pattern']=pattern[:,:,p2,p1].cpu();saved['edge']=(pattern[:,:,p2,p1,None]*v[:,p1]).cpu();saved['factors']={n:x.cpu() for n,x in [('q',q[:,p2]),('q2',q2[:,p2]),('k',k[:,p1]),('k2',k2[:,p1]),('v',v[:,p1])]};return original(q,k,v,q2,k2)
  attn.squared_attention=intercept;h=model.blocks[1].mlp.register_forward_hook(inject)
  def modulecap(m,args,out):saved['module']=out[0][:,p2].cpu()
  hm=attn.register_forward_hook(modulecap)
  try:out,_=run(model,torch.tensor([e['ids']]*4,device='cuda'))
  finally:h.remove();hm.remove();attn.squared_attention=original
  forwards+=4;states[rid]=saved;z=saved['edge'].double().reshape(2,2,9,128);module=saved['module'].double().reshape(2,2,1152)
  for n,t in saved['factors'].items():
   t=t.reshape(2,2,9,128);expected=t[0:1,:,:,:] if n.startswith('q') else t[:,0:1,:,:];independence.append(float((t-expected).abs().max()))
  for i,(a,b) in enumerate(jobs):
   edgeI=z[a,b]-z[a,0]-z[0,b]+z[0,0];heads=torch.stack([edgeI[h]@W[:,h*128:(h+1)*128].T for h in range(9)]);full=module[a,b]-module[a,0]-module[0,b]+module[0,0];patches[rid,a,b]={'heads':heads,'full':full};reconstruction.append({'numerator':float((heads.sum(0)-full).square().sum()),'denominator':float(full.square().sum())});d=(out[i].cpu()-native[rid,0])[sum(e['groups'],[])];site='both_sources' if a==b==1 else 'first_source' if a==1 else 'second_source' if b==1 else None
   if site:capture_replay.append(abs(float(d[0]-d[1])-refidx[rid,site]['margin_change']))
   geometry.extend({'id':rid,'format':e['format'],'item':e['item'],'a':a,'b':b,'head':h,'projected_interaction_squared_norm':float(heads[h].square().sum()),'module_interaction_squared_norm':float(full.square().sum())} for h in range(9))
 print('capture complete',forwards,flush=True);records=[];identity=[];reference_replay=[];edge_full_replay=[]
 for rid in sorted(states):
  e=examples[rid,0];p1,p2,_=src['positions'][rid,0]
  for arm in arms:
   def inject(m,args,out):
    y=out.clone()
    for i,(a,b) in enumerate(jobs):y[i,p1]=writes[rid,a][0].to(y);y[i,p2]=writes[rid,b][1].to(y)
    return y
   def remove(m,args,out):
    if arm=='none':return out
    y=out[0].clone()
    for i,(a,b) in enumerate(jobs):
     p=patches[rid,a,b];v=p['full'] if arm=='full_module' else p['heads'].sum(0) if arm=='all_edges' else p['heads'][int(arm[4:])];y[i,p2]-=v.to(y)
    return(y,out[1])
   hs=[model.blocks[1].mlp.register_forward_hook(inject),attn.register_forward_hook(remove)]
   try:out,_=run(model,torch.tensor([e['ids']]*4,device='cuda'))
   finally:
    for h in hs:h.remove()
   forwards+=4
   for i,(a,b) in enumerate(jobs):
    delta=(out[i].cpu()-native[rid,0])[sum(e['groups'],[])];r={'id':rid,'item':e['item'],'format':e['format'],'a':a,'b':b,'arm':arm,'receiver_index':0,'variant_indices':[0,1],'candidate_logit_changes':delta.tolist()};records.append(r)
    if a==b==0:identity.append(float((out[i].cpu()-native[rid,0]).abs().max()))
  ri={(r['a'],r['b'],r['arm']):torch.tensor(r['candidate_logit_changes']) for r in records if r['id']==rid}
  for a,b in jobs:edge_full_replay.append(float((ri[a,b,'all_edges']-ri[a,b,'full_module']).abs().max()))
  for a,b in jobs:
   if a==0 or b==0:
    for arm in arms:reference_replay.append(float((ri[a,b,arm]-ri[a,b,'none']).abs().max()))
 relative=(sum(x['numerator'] for x in reconstruction)/sum(x['denominator'] for x in reconstruction))**.5;audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'edge_sum_vs_module_interaction_relative_RMS':relative,'Q_vs_first_and_KV_vs_second_source_independence_max_abs':max(independence),'capture_candidate_replay_max_abs':max(capture_replay),'reference_none_full_module_replay_max_abs':max(reference_replay),'all_edges_vs_full_module_candidate_max_abs':max(edge_full_replay),'identity_logits_max_abs':max(identity),'finite':all(math.isfinite(v) for r in records for v in r['candidate_logit_changes']),'whole_model_forwards':forwards,'test_evaluations':0};print(audits,flush=True);assert audits['hashes_unchanged'] and audits['finite'] and relative<1e-4 and max(independence)<1e-5 and max(capture_replay+reference_replay+edge_full_replay+identity)<1e-4
 torch.save({'states':states,'patches':patches},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'geometry':geometry,'competence_source':'H789 both copy formats pass','audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
if __name__=='__main__':main()
