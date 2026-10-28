"""H846 identical-token/position state transfer and frozen early-featurizer tests."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,forward,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_balanced_shared_featurizers import Sparse,BalancedShared
from run_bilin18_early_local_geometry import sha
S='bilin18_same_token_context_transfer'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'examples':ART/'bilin18_candidate_task_screen_captures.pt','location':ART/'bilin18_task_state_location.json','means':ART/'bilin18_operator_state_prediction_captures.pt','fits':ART/'bilin18_balanced_shared_featurizers_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H846','hashes':hashes,'design':'All76 equal-length H843 competent task pairs, both directions, all5 families. At each18 raw preMLP sites transplant all identical-token/index positions, or identical-token positions excluding final position. Token-different positions never edited. Recipient original embeddings and persistent values unchanged.','early_models':'At sites1/2 also exact donor normalizedshape with recipient radius, and all five frozen H836 architectures/both seeds: recipient shape + decoded donor-minus-recipient reconstruction, then normalize and restore recipient radius. Token mean cancels exactly at every edited position, including global fallback. Keep recipient reconstruction error.','interpretation':'Same-token/index difference reflects upstream context. Simultaneous multi-position intervention need not identify a single feature. Reused exploratory task panel; no outcomes-based filtering or retraining. Deeper sweep localizes positive controls; early fits not transported to different sites.','audits':'Manual native runner comparison; all-site identity on first batch; same-token assertion; prefix-before-first-difference state equality; no reserved-test.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');ex=torch.load(paths['examples'],weights_only=False)['examples'];loc=json.loads(paths['location'].read_text());jobs=[r for r in loc['records'] if r['layer']==1 and r['mode']=='complete'];indices=sorted({r['i'] for r in jobs});means=torch.load(paths['means'],weights_only=False)['fits'];states=torch.load(paths['fits'],weights_only=False)['states'];model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 caps={};native={};parts={};recons={};eps=torch.finfo(torch.float32).eps;actual=0;nativeerr=0.;identityerr=0.;prefixerr=0.
 def run(ids,capture=False,patch=None):
  x0=F.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None;cc={}
  for l,block in enumerate(model.blocks):
   raw=block.lambdas[0]*x+block.lambdas[1]*x0;att,bus=block.attn(F.rms_norm(raw,(1152,)),bus);pre=raw+att
   if capture:cc[l]=pre.cpu()
   if patch is not None and patch[0]==l:pre=torch.where(patch[1][:,:,None],patch[2].to(pre),pre)
   x=pre+block.mlp(F.rms_norm(pre,(1152,)))
  return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1]).float()/30),cc
 first=True
 for length in sorted({len(ex[i]['ids']) for i in indices}):
  inds=[i for i in indices if len(ex[i]['ids'])==length]
  for st in range(0,len(inds),16):
   batch=inds[st:st+16];ids=torch.tensor([ex[i]['ids'] for i in batch],device='cuda');y,cc=run(ids,True);actual+=len(batch)
   for k,i in enumerate(batch):native[i]=y[k].double().cpu();caps[i]={l:c[k] for l,c in cc.items()}
   if first:
    ref,_=forward(model,ids);actual+=len(batch);nativeerr=float((ref-y).abs().max())
    for l in range(18):
     yy,_=run(ids,patch=(l,torch.ones_like(ids,dtype=torch.bool),cc[l].cuda()));actual+=len(batch);identityerr=max(identityerr,float((yy-y).abs().max()))
    first=False
 for r in jobs:
  i,j=r['i'],r['j'];firstdiff=next(p for p,(a,b) in enumerate(zip(ex[i]['ids'],ex[j]['ids'])) if a!=b)
  if firstdiff:
   for l in range(18):prefixerr=max(prefixerr,float((caps[i][l][:firstdiff]-caps[j][l][:firstdiff]).abs().max()))
 for l in [1,2]:
  f=means[l];lookup={int(t):k for k,t in enumerate(f['token_ids'])}
  for i in indices:
   b=caps[i][l].double();rad=(b.square().mean(-1,keepdim=True)+eps).sqrt();u=b/rad;tm=torch.stack([f['token_shape_mean'][lookup[t]] if t in lookup else f['shape_mean'] for t in ex[i]['ids']]);parts[l,i]=(u,rad,u-tm)
  for kind in ['scalar48','scalar64','block8','shared_fixed','shared_learned']:
   for seed in [8351,8352]:
    ck=states[l,kind,seed];m=BalancedShared(seed,kind=='shared_fixed').cuda() if kind.startswith('shared') else Sparse(8 if kind=='block8' else 1,48 if kind=='scalar48' else 64).cuda();m.load_state_dict(ck['state_dict'])
    for i in indices:recons[l,kind,seed,i]=(m((parts[l,i][2]/ck['scale']).float().cuda())['reconstruction']*ck['scale']).double().cpu()
    del m
 records=[]
 for l in range(18):
  arms=[('raw',0,'all_same'),('raw',0,'nonfinal_same')]
  if l in [1,2]:arms+=[('shape',0,'all_same')]+[(kind,seed,'all_same') for kind in ['scalar48','scalar64','block8','shared_fixed','shared_learned'] for seed in [8351,8352]]
  for length in sorted({len(ex[r['i']]['ids']) for r in jobs}):
   jj=[r for r in jobs if len(ex[r['i']]['ids'])==length]
   for kind,seed,mode in arms:
    for st in range(0,len(jj),24):
     batch=jj[st:st+24];ids=torch.tensor([ex[r['i']]['ids'] for r in batch],device='cuda');donids=torch.tensor([ex[r['j']]['ids'] for r in batch],device='cuda');mask=ids==donids
     if mode=='nonfinal_same':mask[:,-1]=False
     assert bool((ids[mask]==donids[mask]).all());replace=[]
     for r in batch:
      i,j=r['i'],r['j']
      if kind=='raw':p=caps[j][l].double()
      else:
       ui,rad,ci=parts[l,i];shape=parts[l,j][0] if kind=='shape' else ui+recons[l,kind,seed,j]-recons[l,kind,seed,i];p=shape/shape.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-30)*(rad.square()-eps).clamp_min(0).sqrt()
      replace.append(p)
     y,_=run(ids,patch=(l,mask,torch.stack(replace).cuda()));actual+=len(batch)
     for k,r in enumerate(batch):
      i,j=r['i'],r['j'];base=measure(native[i],ex[i]['groups'])['margin'];value=measure(y[k],ex[i]['groups'])['margin'];sgn=1 if r['recipient']==1 else -1;lp=native[i].log_softmax(0);records.append({key:r[key] for key in ['id','family','split','recipient','donor','i','j','native_answer_gap']}|{'layer':l,'kind':kind,'seed':seed,'mode':mode,'positions_edited':int(mask[k].sum()),'donor_directed_delta':sgn*(value-base),'KL':float((lp.exp()*(lp-y[k].double().cpu().log_softmax(0))).sum())})
  print('DONE',l,'records',len(records),flush=True)
 summary=[]
 for l,kind,seed,mode,fam in sorted({(r['layer'],r['kind'],r['seed'],r['mode'],r['family']) for r in records}):
  rr=[r for r in records if (r['layer'],r['kind'],r['seed'],r['mode'],r['family'])==(l,kind,seed,mode,fam)];summary.append({'layer':l,'kind':kind,'seed':seed,'mode':mode,'family':fam,'gap_fraction':sum(r['donor_directed_delta'] for r in rr)/sum(r['native_answer_gap'] for r in rr),'mean_KL':float(np.mean([r['KL'] for r in rr]))})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_logits_max_abs':nativeerr,'identity_logits_max_abs':identityerr,'identical_prefix_state_max_abs':prefixerr,'same_token_position_assertions':True,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['KL']) for r in records)};assert audits['hashes_unchanged'] and max(nativeerr,identityerr,prefixerr)<1e-4 and audits['finite'];torch.save({'premlp':caps,'native':native,'parts':parts,'reconstructions':recons},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
