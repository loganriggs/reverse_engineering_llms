"""H859 frozen single-state calibration, document-disjoint natural output confirmation."""
import json,time,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_balanced_shared_featurizers import Sparse,BalancedShared
from run_bilin18_relay_contrast_subspaces import run
from run_bilin18_early_local_geometry import sha
S='bilin18_norm_frozen_confirmation'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'fits':ART/'bilin18_relay_natural_featurizers_captures.pt','means':ART/'bilin18_relay_natural_data_captures.pt','panel':ART/'bilin18_frozen_featurizer_confirmation_captures.pt','manifest':ART/'bilin18_frozen_featurizer_confirmation_manifest.json','original_manifest':ART/'bilin18_early_bsf_data_manifest.json','development_manifest':ART/'bilin18_fresh_early_geometry_panel_manifest.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};manifest=json.loads(paths['manifest'].read_text());old=json.loads(paths['original_manifest'].read_text());dev=json.loads(paths['development_manifest'].read_text());exclude={r['document_hash'] for r in old['records']+dev['records']};disjoint=not bool(exclude&{r['document_hash'] for r in manifest['records']});assert disjoint;panel=torch.load(paths['panel'],weights_only=False);ids=panel['all_token_ids'].clone();positions=panel['positions'];del panel;assert len(ids)==512 and all(hashlib.sha256(row.numpy().astype(np.int64).tobytes()).hexdigest()==rec['window_sha256_int64'] for row,rec in zip(ids,manifest['records']));cfg={'experiment':'H859','hashes':hashes,'design':'All512 H837 documents/locked positions, document-disjoint from allH6812560 andH8241024 including H851training/development. Every10 H852fit and all4 H857single-state inference variants retained. No tuning, model/seed selection or taskdirection load.','freshness':'New to L4 featurizer fitting/selection and norm-rule natural-output evaluation; reused broader research documents previously evaluated at other sites. Not pretraining-disjoint or entirely untouched research data.','metrics':'FullrawpreMLP4 reconstruction replacement at selectedposition; KL(native||replacement),TV,nexttokenCE when followingtoken exists in128window. Seen/unseen trainingtoken mean strata and per-document changes retained.','rule':'Same H857 independentstate bias-centered residualnorm retention, learned or frozen-support LS coordinates, rawRMS restored perinput. No clipping/gain fitting.','controls':'Native prefix-truncated capture and exactstate identity replacement, manifest/window/source hash checks, float64 LS/norm/rank audits. Original reservedtest never evaluated.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');states=torch.load(paths['fits'],weights_only=False)['states'];f=torch.load(paths['means'],weights_only=False)['fits'][4];lookup={int(t):i for i,t in enumerate(f['token_ids'])};tids=[int(ids[i,p]) for i,p in enumerate(positions)];seen=[t in lookup for t in tids];model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 rawcaps=torch.empty(512,1152);native={};actual=0;ident=0.
 for pos in sorted(set(positions)):
  ix=[i for i,p in enumerate(positions) if p==pos]
  for st in range(0,len(ix),8):
   batch=ix[st:st+8];tokens=ids[batch,:pos+1].cuda();x0=F.rms_norm(model.embedding(tokens),(1152,));x=x0;bus=None
   for l,block in enumerate(model.blocks):
    raw=block.lambdas[0]*x+block.lambdas[1]*x0;att,bus=block.attn(F.rms_norm(raw,(1152,)),bus);pre=raw+att
    if l==4:rawcaps[batch]=pre[:,-1].cpu()
    x=pre+block.mlp(F.rms_norm(pre,(1152,)))
   y=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1]).float()/30);actual+=len(batch);pp=torch.full((len(batch),),pos,dtype=torch.long,device='cuda');yy=run(model,tokens,rawcaps[batch].cuda(),pp);actual+=len(batch);ident=max(ident,float((yy-y).abs().max()))
   for k,i in enumerate(batch):native[i]=y[k].double().cpu()
 print('CAPTURED512',flush=True);b=rawcaps.double();eps=torch.finfo(torch.float32).eps;rad=(b.square().mean(1,keepdim=True)+eps).sqrt();u=b/rad;tm=torch.stack([f['token_shape_mean'][lookup[t]] if t in lookup else f['shape_mean'] for t in tids]);res=u-tm;predictions={};gains=[];normerr=0.;lsworse=0.;ortherr=0.
 for kind in ['scalar48','scalar64','block8','shared_fixed','shared_learned']:
  for seed in [8351,8352]:
   name=f'{kind}_{seed}';ck=states[4,kind,seed];m=BalancedShared(seed,kind=='shared_fixed').cuda() if kind.startswith('shared') else Sparse(8 if kind=='block8' else 1,48 if kind=='scalar48' else 64).cuda();m.load_state_dict(ck['state_dict']);outs=[m(z) for z in (res/ck['scale']).float().cuda().split(128)];learned=torch.cat([o['reconstruction'] for o in outs]).double().cpu()*ck['scale'];support=torch.cat([o['support'] for o in outs]).bool().cpu();D=(m.dictionary_matrix().T if kind.startswith('shared') else m.decoder).double().cpu();beta=m.bias.double().cpu()*ck['scale'];cache={};ls=[]
   for i in range(512):
    key=tuple(support[i].nonzero().flatten().tolist())
    if key not in cache:
     _,sv,v=torch.linalg.svd(D[list(key)],full_matrices=False);rank=int((sv>sv[0]*1e-10).sum());Q=v[:rank].T;cache[key]=Q;ortherr=max(ortherr,float((Q.T@Q-torch.eye(rank)).abs().max()))
    Q=cache[key];ls.append((res[i]-beta)@Q@Q.T+beta)
   ls=torch.stack(ls);lsworse=max(lsworse,float(((ls-res).square().sum(1)-(learned-res).square().sum(1)).max()));variants={'learned':learned,'ls':ls}
   for base,p in list(variants.items()):
    den=(p-beta).norm(dim=1,keepdim=True);assert float(den.min())>1e-10;gain=(res-beta).norm(dim=1,keepdim=True)/den;pn=beta+(p-beta)*gain;variants[base+'_norm']=pn;normerr=max(normerr,float(((pn-beta).norm(dim=1)-(res-beta).norm(dim=1)).abs().max()));gains.append({'model':name,'base':base,'mean':float(gain.mean()),'max':float(gain.max())})
   for variant,p in variants.items():shape=tm+p;predictions[name,variant]=shape/shape.square().mean(1,keepdim=True).sqrt().clamp_min(1e-30)*(rad.square()-eps).sqrt()
   del m,outs
   print('ENCODED',name,flush=True)
 records=[]
 for pos in sorted(set(positions)):
  ix=[i for i,p in enumerate(positions) if p==pos]
  for st in range(0,len(ix),8):
   batch=ix[st:st+8];tokens=ids[batch,:pos+1].cuda();pp=torch.full((len(batch),),pos,dtype=torch.long,device='cuda');lp=torch.stack([native[i] for i in batch]).log_softmax(-1)
   for (name,variant),pred in predictions.items():
    y=run(model,tokens,pred[batch].cuda(),pp);actual+=len(batch);lq=y.double().cpu().log_softmax(-1);kl=(lp.exp()*(lp-lq)).sum(-1);tv=(lp.exp()-lq.exp()).abs().sum(-1)/2
    for k,i in enumerate(batch):label=int(ids[i,pos+1]) if pos<127 else None;records.append({'document_index':i,'document_hash':manifest['records'][i]['document_hash'],'position':pos,'seen':seen[i],'model':name,'variant':variant,'KL':float(kl[k]),'TV':float(tv[k]),'next_token_CE_change':float(lp[k,label]-lq[k,label]) if label is not None else None})
  print('DONE',pos,flush=True)
  (ART/f'{S}_partial.json').write_text(json.dumps({'records':records,'status':'running'}))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'document_disjoint_from_training_development':disjoint,'panel_window_hashes_match':True,'seen_tokens':sum(seen),'unseen_tokens':512-sum(seen),'identity_logits_max_abs':ident,'SVD_orthogonality_max_abs':ortherr,'residual_norm_retention_max_abs':normerr,'LS_error_increase_max':lsworse,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['KL']) for r in records)};assert audits['hashes_unchanged'] and ident<1e-4 and max(ortherr,normerr)<1e-9 and lsworse<1e-4 and audits['finite'];torch.save({'raw_premlp':rawcaps,'native':native,'predictions':predictions,'seen':seen},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'gains':gains,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
