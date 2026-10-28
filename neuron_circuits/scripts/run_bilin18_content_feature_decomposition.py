"""H845 token mean/context/radius and frozen featurizers at causal content positions."""
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,forward,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_balanced_shared_featurizers import Sparse,BalancedShared
from run_bilin18_early_local_geometry import sha
S='bilin18_content_feature_decomposition'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'examples':ART/'bilin18_candidate_task_screen_captures.pt','location':ART/'bilin18_task_state_location.json','means':ART/'bilin18_operator_state_prediction_captures.pt','fits':ART/'bilin18_balanced_shared_featurizers_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H845','hashes':hashes,'design':'All31 equal-length agreement/sentiment pairs from H843, both directions, sites1/2 raw preMLP state. Replace only changed-token positions, before both MLP and residual skip. Recipient value bus/x0 remain native.','factorial':'Normalizedshape u=raw/sqrt(mean(raw²)+eps), M=frozen training token mean, C=u-M, radius r. All8 donor/recipient combinations of M,C,r. Mixed shape renormalized and restored to raw RMS. Context means mean-centered remainder, not proven purely contextual content.','featurizers':'All five H836 models, both seeds, no refit: replace C donor-minus-recipient with decoded reconstruction difference, preserving recipient reconstruction error, with recipient/donor tokenmean; recipient radius fixed. Compare with exact C and M+C arms. Unit dictionary; signed scalars, non-Grassmannian blocks, shared union.','controls':'Native runner equality, identity reconstructed raw input, full raw donor replacement versus allfactor reconstruction. No reserved-test use. Reused task panel and trained tokenmeans; unknown tokens use globalmean and reported.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');ex=torch.load(paths['examples'],weights_only=False)['examples'];loc=json.loads(paths['location'].read_text());jobs=[r for r in loc['records'] if r['layer']==1 and r['mode']=='complete' and r['family'] in ['agreement','sentiment']];indices=sorted({r['i'] for r in jobs});means=torch.load(paths['means'],weights_only=False)['fits'];states=torch.load(paths['fits'],weights_only=False)['states'];model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 caps={};native={};eps=torch.finfo(torch.float32).eps;actual=0;nativeerr=0.;identerr=0.;fullerr=0.;unknown={};parts={};recons={}
 def run(ids,capture=False,patch=None):
  x0=F.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None;cc={}
  for l,block in enumerate(model.blocks):
   raw=block.lambdas[0]*x+block.lambdas[1]*x0;att,bus=block.attn(F.rms_norm(raw,(1152,)),bus);pre=raw+att
   if capture and l in [1,2]:cc[l]=pre.cpu()
   if patch is not None and patch[0]==l:pre=torch.where(patch[1][:,:,None],patch[2].to(pre),pre)
   x=pre+block.mlp(F.rms_norm(pre,(1152,)))
  return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1]).float()/30),cc
 for length in sorted({len(ex[i]['ids']) for i in indices}):
  inds=[i for i in indices if len(ex[i]['ids'])==length]
  for st in range(0,len(inds),16):
   batch=inds[st:st+16];ids=torch.tensor([ex[i]['ids'] for i in batch],device='cuda');y,cc=run(ids,True);ref,_=forward(model,ids);actual+=2*len(batch);nativeerr=max(nativeerr,float((y-ref).abs().max()))
   for k,i in enumerate(batch):native[i]=y[k].double().cpu();caps[i]={l:c[k] for l,c in cc.items()}
 for l in [1,2]:
  f=means[l];lookup={int(t):i for i,t in enumerate(f['token_ids'])};unknown[l]=0
  for i in indices:
   b=caps[i][l].double();rad=(b.square().mean(-1,keepdim=True)+eps).sqrt();u=b/rad;tm=torch.stack([f['token_shape_mean'][lookup[t]] if t in lookup else f['shape_mean'] for t in ex[i]['ids']]);unknown[l]+=sum(t not in lookup for t in ex[i]['ids']);parts[l,i]=(tm,u-tm,rad)
  for kind in ['scalar48','scalar64','block8','shared_fixed','shared_learned']:
   for seed in [8351,8352]:
    ck=states[l,kind,seed];m=BalancedShared(seed,kind=='shared_fixed').cuda() if kind.startswith('shared') else Sparse(8 if kind=='block8' else 1,48 if kind=='scalar48' else 64).cuda();m.load_state_dict(ck['state_dict'])
    for i in indices:recons[l,kind,seed,i]=(m((parts[l,i][1]/ck['scale']).float().cuda())['reconstruction']*ck['scale']).double().cpu()
    del m
 records=[]
 for l in [1,2]:
  for length in sorted({len(ex[r['i']]['ids']) for r in jobs}):
   jj=[r for r in jobs if len(ex[r['i']]['ids'])==length]
   arms=[('exact',0,mask) for mask in range(8)]+[('raw',0,7)]+[(kind,seed,mask) for kind in ['scalar48','scalar64','block8','shared_fixed','shared_learned'] for seed in [8351,8352] for mask in [2,3]]
   for kind,seed,mask in arms:
    for st in range(0,len(jj),24):
     batch=jj[st:st+24];ids=torch.tensor([ex[r['i']]['ids'] for r in batch],device='cuda');donids=torch.tensor([ex[r['j']]['ids'] for r in batch],device='cuda');changed=ids!=donids;replace=[]
     for r in batch:
      i,j=r['i'],r['j'];mi,ci,ri=parts[l,i];mj,cj,rj=parts[l,j]
      if kind=='raw':p=caps[j][l].double()
      else:
       mm=mj if mask&1 else mi;rad=rj if mask&4 else ri
       c=(cj if mask&2 else ci) if kind=='exact' else ci+recons[l,kind,seed,j]-recons[l,kind,seed,i];shape=mm+c;p=shape/shape.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-30)*(rad.square()-eps).clamp_min(0).sqrt()
      replace.append(p)
     y,_=run(ids,patch=(l,changed,torch.stack(replace).cuda()));actual+=len(batch)
     for k,r in enumerate(batch):
      i,j=r['i'],r['j'];base=measure(native[i],ex[i]['groups'])['margin'];value=measure(y[k],ex[i]['groups'])['margin'];sgn=1 if r['recipient']==1 else -1;lp=native[i].log_softmax(0);records.append({key:r[key] for key in ['id','family','split','recipient','donor','i','j','native_answer_gap']}|{'layer':l,'kind':kind,'seed':seed,'mask':mask,'donor_directed_delta':sgn*(value-base),'KL':float((lp.exp()*(lp-y[k].double().cpu().log_softmax(0))).sum())})
      if kind=='exact' and mask==0:identerr=max(identerr,float((y[k].double().cpu()-native[i]).abs().max()))
   print('DONE',l,length,flush=True)
 rr={(r['layer'],r['i'],r['kind'],r['mask']):r for r in records if r['kind'] in ['raw','exact']};fullerr=max(abs(r['donor_directed_delta']-rr[r['layer'],r['i'],'exact',7]['donor_directed_delta']) for r in records if r['kind']=='raw');summary=[]
 for l,kind,seed,mask,fam in sorted({(r['layer'],r['kind'],r['seed'],r['mask'],r['family']) for r in records}):
  rows=[r for r in records if (r['layer'],r['kind'],r['seed'],r['mask'],r['family'])==(l,kind,seed,mask,fam)];summary.append({'layer':l,'kind':kind,'seed':seed,'mask':mask,'family':fam,'gap_fraction':sum(r['donor_directed_delta'] for r in rows)/sum(r['native_answer_gap'] for r in rows),'mean_KL':float(np.mean([r['KL'] for r in rows]))})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_logits_max_abs':nativeerr,'identity_logits_max_abs':identerr,'full_decomposition_vs_raw_margin_max_abs':fullerr,'whole_model_forwards':actual,'unknown_token_occurrences':unknown,'test_evaluations':0,'finite':all(np.isfinite(r['KL']) for r in records)};assert audits['hashes_unchanged'] and max(nativeerr,identerr,fullerr)<1e-4 and audits['finite'];torch.save({'premlp':caps,'native':native,'parts':parts,'reconstructions':recons},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
