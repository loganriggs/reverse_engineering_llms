"""H685: lexical/position controls and exploratory atlas for stable early blocks."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_early_bsf_extended import BSF
STEM='bilin18_bsf_lexical_audit'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_early_bsf_data_captures.pt','stability':ART/'bilin18_early_bsf_stability.json','trainer':ROOT/'scripts/run_bilin18_early_bsf_extended.py','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);stab=json.loads(paths['stability'].read_text());cfg=dict(experiment='H685',hashes=hashes,design='Value0/residual0/residual1 width16 seed681. Top8blocks per site ranked by minimum(overlap*supportJaccard) across two H684 seed matches, require32validation activations. Training-only smoothed token/frequency/position support predictors; validation Brier skill and within-token coordinate variation. Exploratory snippets from validation, test untouched.',limitations='Block selection and descriptive evaluation reuse validation. Frequency/position predictors are simple controls, not exhaustive confound adjustment. Residual variation is not proof of semantic manifold. Token identities may reflect tokenization/lexical categories.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);trids=data['token_ids'][:2048].flatten().cuda();vaids=data['token_ids'][2048:2304].flatten().cuda();trpos=torch.arange(16,device='cuda').repeat(2048);vapos=torch.arange(16,device='cuda').repeat(256);tc=torch.bincount(trids,minlength=50257).float();vc=torch.bincount(vaids,minlength=50257).float();freq=(tc.clamp_min(1).log2().floor().long()).clamp_max(15);records=[];checkpoint_hashes={}
 for site in ['value0','residual0','residual1']:
  p=ART/'early_bsf_extended'/f'{site}_r16_seed681.pt';checkpoint_hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();ck=torch.load(p,weights_only=False);m=BSF(1152,16).cuda();m.load_state_dict(ck['state_dict']);sr=[x for x in stab['records'] if x['site']==site and x['block_dimension']==16];scores=np.minimum(*[np.array(x['learned']['overlap'])*np.array(x['learned']['jaccard']) for x in sr]);eligible=np.array(ck['record']['geometry']['occupancy'])>=32;scores[~eligible]=-1;chosen=np.argsort(-scores)[:8].tolist();assert min(scores[chosen])>=0
  codes=[];masks=[]
  for lo,hi in [(0,2048),(2048,2304)]:
   x=(data['activations'][site][lo:hi].reshape(-1,1152).cuda()-ck['mean'].cuda())/ck['scale'].cuda();zs=[];ms=[]
   for b in x.split(512):z,mask=m.encode(b);zs.append(z[:,chosen]);ms.append(mask[:,chosen])
   codes.append(torch.cat(zs));masks.append(torch.cat(ms))
  zt,zv=codes;yt,yv=masks;prior=yt.mean(0);token_sum=torch.zeros(50257,8,device='cuda').index_add_(0,trids,yt);tokenpred=(token_sum+8*prior)/(tc[:,None]+8);pos_sum=torch.zeros(16,8,device='cuda').index_add_(0,trpos,yt);pospred=(pos_sum+8*prior)/(2048+8);fc=torch.bincount(freq[trids],minlength=16).float();fs=torch.zeros(16,8,device='cuda').index_add_(0,freq[trids],yt);fpred=(fs+8*prior)/(fc[:,None]+8)
  preds={'token':tokenpred[vaids],'position':pospred[vapos],'frequency':fpred[freq[vaids]]};logit=lambda p:torch.logit(p.clamp(.0001,.9999));preds['token_plus_position']=(logit(preds['token'])+logit(preds['position'])-logit(prior)).sigmoid();den=(yv-prior).square().mean(0);skills={name:1-(pred-yv).square().mean(0)/den.clamp_min(1e-12) for name,pred in preds.items()}
  for b,block in enumerate(chosen):
   active=yt[:,b].bool();ids=trids[active];zz=zt[active,b];counts=torch.bincount(ids,minlength=50257).float();sums=torch.zeros(50257,16,device='cuda').index_add_(0,ids,zz);means=sums/counts.clamp_min(1)[:,None];repeat=counts[ids]>=8;zr=zz[repeat];idr=ids[repeat];within=float((zr-means[idr]).square().sum()/(zr-zr.mean(0)).square().sum()) if len(zr)>1 else None
   unit=zz/zz.norm(dim=-1,keepdim=True).clamp_min(1e-12);usum=torch.zeros(50257,16,device='cuda').index_add_(0,ids,unit);um=usum/counts.clamp_min(1)[:,None];ur=unit[repeat];angular=float((ur-um[idr]).square().sum()/(ur-ur.mean(0)).square().sum().clamp_min(1e-12)) if len(ur)>1 else None
   vsum=torch.zeros(50257,device='cuda').index_add_(0,vaids,yv[:,b]);rep=vc>=4;mixed=rep&(vsum>0)&(vsum<vc);actcounts=torch.bincount(vaids[yv[:,b].bool()],minlength=50257);top=actcounts.topk(8);top_tokens=[dict(token=tok.decode([int(t)]),count=int(n)) for n,t in zip(top.values,top.indices) if n>0];examples=[];useddocs=set();usedtokens={}
   for k in zv[:,b].norm(dim=-1).argsort(descending=True).cpu().tolist():
    if not yv[k,b]:continue
    doc=k//16;tid=int(vaids[k]);pos=data['positions'][k%16]
    if doc in useddocs or usedtokens.get(tid,0)>=2:continue
    useddocs.add(doc);usedtokens[tid]=usedtokens.get(tid,0)+1;context=data['all_token_ids'][2048+doc];examples.append(dict(validation_document=doc,position=pos,token=tok.decode([tid]),prefix=tok.decode(context[max(0,pos-24):pos+1].tolist()),suffix=tok.decode(context[pos+1:min(128,pos+9)].tolist())))
    if len(examples)==6:break
   records.append(dict(site=site,block=block,stability_product_min=float(scores[block]),validation_active=int(yv[:,b].sum()),training_active=int(active.sum()),brier_skill={name:float(v[b]) for name,v in skills.items()},within_token_coordinate_variance_fraction=within,within_token_unit_coordinate_variance_fraction=angular,within_token_eligible_active_examples=int(repeat.sum()),repeated_validation_token_types=int(rep.sum()),mixed_support_repeated_token_types=int(mixed.sum()),top_active_tokens=top_tokens,examples=examples))
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},checkpoints_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in checkpoint_hashes.items()),blocks=len(records),test_evaluations=0,whole_model_forwards=0,finite=all(np.isfinite(list(x['brier_skill'].values())).all() for x in records));(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,checkpoint_hashes=checkpoint_hashes,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits)
if __name__=='__main__':main()
