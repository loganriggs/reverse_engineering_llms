"""H795 reconstruct and causally remove first-to-second source attention interactions."""
import json,time,math
from pathlib import Path
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
S='bilin18_source_edge_heads'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'captures':ART/'bilin18_source_donor_specificity_captures.pt','early':ART/'bilin18_source_interaction_trace_early_captures.pt','reference':ART/'bilin18_early_interaction_removal.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};c=torch.load(paths['captures'],weights_only=False);early=torch.load(paths['early'],weights_only=False);ref=json.loads(paths['reference'].read_text());examples=c['examples'];native=c['native'];writes=c['writes'];words=ref['candidate_words'];arms=['none','all_edges','full_module']+[f'head{h}' for h in range(9)];cfg={'experiment':'H795','hashes':hashes,'arms':arms,'design':'Replay source factorial; capture actual L2 post-normalization/RoPE Q,K,Q2,K2 and mixedV. Reconstruct first-source→second-source signed edge per head, project factorial interaction through c_proj. Remove one head interaction, all edge interactions, or full-module interaction at second-source output.','prediction':'First-to-second source edge accounts for the L2 second-source factorial interaction. Individual heads may differ in causal output effect; test all9, not only large-norm heads.','limitations':'Head decomposition under this parametrization; nonlinear downstream effects need not sum. Reused panel. Removal is a local hybrid intervention, not unique mediation.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 attn=model.blocks[2].attn;assert attn.squared_attn;original=attn.squared_attention;W=attn.c_proj.weight.cpu().double();states={};patches={};geometry=[];reconstruction=[];independence=[];capture_replay=[];forwards=0;refidx={(r['id'],r['a'],r['b'],r['arm']):r for r in ref['records']};jobs=[(a,b) for a in range(4) for b in range(4)]
 for rid in sorted({k[0] for k in examples}):
  e=examples[rid,0];p1,p2,_=e['positions'];saved={}
  def inject(m,args,out):
   y=out.clone()
   for i,(a,b) in enumerate(jobs):y[i,p1]=writes[rid,a][0].to(y);y[i,p2]=writes[rid,b][1].to(y)
   return y
  def intercept(q,k,v,q2,k2):
   pattern=(torch.einsum('bqhd,bkhd->bhqk',q,k)/128)*(torch.einsum('bqhd,bkhd->bhqk',q2,k2)/128);saved['edge']=(pattern[:,:,p2,p1,None]*v[:,p1]).cpu();saved['factors']={n:x.cpu() for n,x in [('q',q[:,p2]),('q2',q2[:,p2]),('k',k[:,p1]),('k2',k2[:,p1]),('v',v[:,p1])]};return original(q,k,v,q2,k2)
  attn.squared_attention=intercept;h=model.blocks[1].mlp.register_forward_hook(inject)
  try:out,_=run(model,torch.tensor([e['ids']]*16,device='cuda'))
  finally:h.remove();attn.squared_attention=original
  forwards+=16;states[rid]=saved;z=saved['edge'].double().reshape(4,4,9,128);module=early[rid][2,'attn_output'].double().reshape(4,4,3,1152)[:,:,1]
  for n,t in saved['factors'].items():
   t=t.reshape(4,4,9,128);expected=t[0:1,:,:,:] if n.startswith('q') else t[:,0:1,:,:];independence.append(float((t-expected).abs().max()))
  for i,(a,b) in enumerate(jobs):
   edgeI=z[a,b]-z[a,0]-z[0,b]+z[0,0];heads=torch.stack([edgeI[h]@W[:,h*128:(h+1)*128].T for h in range(9)]);full=module[a,b]-module[a,0]-module[0,b]+module[0,0];patches[rid,a,b]={'heads':heads,'full':full};reconstruction.append({'numerator':float((heads.sum(0)-full).square().sum()),'denominator':float(full.square().sum())});d=(out[i].cpu()-native[rid,0])[e['candidate_ids']];capture_replay.append(float((d-torch.tensor(refidx[rid,a,b,'none']['candidate_logit_changes'])).abs().max()));geometry.extend({'id':rid,'format':e['format'],'item':e['item'],'a':a,'b':b,'head':h,'projected_interaction_squared_norm':float(heads[h].square().sum()),'module_interaction_squared_norm':float(full.square().sum())} for h in range(9))
 print('capture complete',forwards,flush=True);records=[];identity=[];reference_replay=[];edge_full_replay=[]
 for rid in sorted(states):
  e=examples[rid,0];p1,p2,_=e['positions']
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
   try:out,_=run(model,torch.tensor([e['ids']]*16,device='cuda'))
   finally:
    for h in hs:h.remove()
   forwards+=16
   for i,(a,b) in enumerate(jobs):
    delta=(out[i].cpu()-native[rid,0])[e['candidate_ids']];r={'id':rid,'item':e['item'],'format':e['format'],'a':a,'b':b,'arm':arm,'receiver_index':e['answer_index'],'variant_indices':[examples[rid,v]['answer_index'] for v in range(4)],'candidate_logit_changes':delta.tolist()};records.append(r)
    if a==b==0:identity.append(float((out[i].cpu()-native[rid,0]).abs().max()))
    if arm in ['none','full_module','all_edges']:
     target=refidx[rid,a,b,'none' if arm=='none' else 'source_attention'];err=float((delta-torch.tensor(target['candidate_logit_changes'])).abs().max());(edge_full_replay if arm=='all_edges' else reference_replay).append(err)
 relative=(sum(x['numerator'] for x in reconstruction)/sum(x['denominator'] for x in reconstruction))**.5;audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'edge_sum_vs_module_interaction_relative_RMS':relative,'Q_vs_first_and_KV_vs_second_source_independence_max_abs':max(independence),'capture_candidate_replay_max_abs':max(capture_replay),'reference_none_full_module_replay_max_abs':max(reference_replay),'all_edges_vs_H794_module_clamp_candidate_max_abs':max(edge_full_replay),'identity_logits_max_abs':max(identity),'finite':all(math.isfinite(v) for r in records for v in r['candidate_logit_changes']),'whole_model_forwards':forwards,'test_evaluations':0};print(audits,flush=True);assert audits['hashes_unchanged'] and audits['finite'] and relative<1e-4 and max(independence)<1e-5 and max(capture_replay+reference_replay+edge_full_replay+identity)<1e-4
 torch.save({'states':states,'patches':patches},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'geometry':geometry,'competence':ref['competence'],'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
if __name__=='__main__':main()
