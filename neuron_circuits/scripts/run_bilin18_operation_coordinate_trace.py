"""H750 exact native residual contribution trace into verified Q2/K2 coordinates."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_operation_coordinate_trace'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time()
 paths={'capture':ART/'bilin18_fresh_wording_subspace_effects_captures.pt','results':ART/'bilin18_fresh_wording_subspace_effects.json','bases':ART/'bilin18_operation_nuisance_subspaces_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py')};hashes={k:sha(v) for k,v in paths.items()}
 cfg={'experiment':'H750','hashes':hashes,'design':'Exact native residual expansion at L13 attention input into embedding/skip injection plus L0..12 attention and MLP outputs, with learned residual multipliers. Project through L13H0 Q2/K2 weights and observed native RMS denominators into frozen H742 rank4 coordinates. Pairwise binding/copy differences are decomposed without fitting. Attribution shares use sum dot(component_difference,total_difference)/sum squared total_difference; can be negative or >1. All H747 cells, native competence carried forward.','limitations':'Native-state algebraic attribution with context-specific RMS factors, not causal mediation. Earlier outputs include their upstream dependencies, so a large direct contribution need not be earliest origin.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 cap=torch.load(paths['capture'],weights_only=False);old=json.loads(paths['results'].read_text());bases=torch.load(paths['bases'],weights_only=False)['bases'];examples=cap['examples'];meta={(r['id'],r['receiver']):r for r in old['records'] if r['scope']=='native'}
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 lambdas=[b.lambdas.detach().double().cpu().tolist() for b in model.blocks[:14]];embedw=1.
 for a,b in lambdas:embedw=a*embedw+b
 weights={}
 for j in range(13):
  w=1.
  for a,b in lambdas[j+1:14]:w*=a
  for kind in ['attn','mlp']:weights[f'L{j}_{kind}']=w
 names=['embedding']+list(weights);coordinates={};contributions={};residualerr=[];coorderr=[];logiterr=[];buckets=defaultdict(list)
 for key,e in examples.items():buckets[len(e['ids'])].append(key)
 for length,keys in buckets.items():
  for lo in range(0,len(keys),12):
   bb=keys[lo:lo+12];store={};hooks=[]
   def select(t):return torch.stack([t[i,[length-1,examples[k]['source_position']]] for i,k in enumerate(bb)]).detach().cpu().double()
   def b0(m,args):store['x0']=select(args[2])
   hooks.append(model.blocks[0].register_forward_pre_hook(b0))
   for j in range(13):
    for kind in ['attn','mlp']:
     def h(m,args,out,j=j,kind=kind):store[f'L{j}_{kind}']=select(out[0] if kind=='attn' else out)
     hooks.append(getattr(model.blocks[j],kind).register_forward_hook(h))
   def b13(m,args):store['b13']=select(m.lambdas[0]*args[0]+m.lambdas[1]*args[2])
   hooks.append(model.blocks[13].register_forward_pre_hook(b13))
   for c in ['q2','k2']:
    def raw(m,args,out,c=c):store[c]=select(out[:,:,:128])
    hooks.append(getattr(model.blocks[13].attn,'c_'+c).register_forward_hook(raw))
   try:logits,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:
    for h in hooks:h.remove()
   components=torch.stack([embedw*store['x0']]+[weights[n]*store[n] for n in weights],dim=1);total=components.sum(1);residualerr.extend(((total-store['b13']).norm(dim=-1)/store['b13'].norm(dim=-1)).flatten().tolist());eps=torch.finfo(torch.float32).eps
   for c,pos,u in [('q2',0,'uq2'),('k2',1,'uk2')]:
    W=getattr(model.blocks[13].attn,'c_'+c).weight[:128].detach().cpu().double();B=bases['frozen_H742',c];rmsb=(store['b13'][:,pos].square().mean(-1)+eps).sqrt();rmsq=(store[c][:,pos].square().mean(-1)+eps).sqrt();cc=(components[:,:,pos]@W.T@B)/(rmsb*rmsq)[:,None,None]
    for i,k in enumerate(bb):
     expected=cap['states'][k][u]@B;coorderr.append(float((cc[i].sum(0)-expected).abs().max()));coordinates[k,c]=expected;contributions[k,c]=cc[i]
   for i,k in enumerate(bb):logiterr.append(float((logits[i].cpu()-cap['native_logits'][k]).abs().max()))
 records=[];groups=defaultdict(list)
 for key,donor in cap['pairs']:
  m=meta[key]
  if m['family']!='entity_binding':continue
  cell='|'.join(m[k] for k in ['name_set','wording','format']);competent=all(old['competence'][cell+'|'+fam]['pass'] for fam in ['entity_binding','sequence_copy'])
  for c in ['q2','k2']:
   delta=coordinates[donor,c]-coordinates[key,c];parts=contributions[donor,c]-contributions[key,c];den=float(delta.square().sum());rec={'id':key[0],'receiver':key[1],'cell':cell,'competent':competent,'component':c,'total_energy':den,'alignment':(parts@delta).tolist(),'part_energy':parts.square().sum(-1).tolist()};records.append(rec);groups[competent,m['format'],c].append(rec)
 summary=[]
 for (competent,fmt,c),rr in groups.items():
  den=sum(r['total_energy'] for r in rr);shares=[sum(r['alignment'][i] for r in rr)/den for i in range(len(names))];summary.append({'competent':competent,'format':fmt,'component':c,'pairs':len(rr),'alignment_share_sum':sum(shares),'shares':dict(zip(names,shares))})
 audits={'hashes_unchanged':{k:sha(v)==hashes[k] for k,v in paths.items()},'residual_reconstruction_max_relative':max(residualerr),'coordinate_reconstruction_max_abs':max(coorderr),'native_logit_replay_max_abs':max(logiterr),'alignment_share_sum_max_error':max(abs(s['alignment_share_sum']-1) for s in summary),'whole_model_forwards':len(examples),'test_evaluations':0};print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and max(residualerr)<1e-5 and max(coorderr)<1e-4 and max(logiterr)<1e-4 and audits['alignment_share_sum_max_error']<1e-5
 torch.save({'coordinates':coordinates,'contributions':contributions,'names':names,'lambdas':lambdas,'weights':weights,'embedding_weight':embedw},ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'summary':summary,'names':names,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
 for s in summary:print(s['competent'],s['format'],s['component'],sorted(s['shares'].items(),key=lambda kv:abs(kv[1]),reverse=True)[:10],flush=True)
if __name__=='__main__':main()
