"""H998 bounded natural-text residual pilot using only historical BSF train documents."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward
from audit_bilin18_mlp_discriminators import TOKENIZER
from transformers import AutoTokenizer
from pathlib import Path
from collections import Counter
import torch,torch.nn.functional as F,numpy as np,json,hashlib,time
S='bilin18_natural_residual_discovery'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');mp=ART/'bilin18_early_bsf_data_manifest.json';manifest=json.loads(mp.read_text());eligible=[r for r in manifest['records'] if r['split']=='train'];eligible.sort(key=lambda r:hashlib.sha256(('H998:'+r['document_hash']).encode()).hexdigest());selected=eligible[:256];assert len({r['document_hash'] for r in selected})==256;reserved={r['document_hash'] for r in manifest['records'] if r['split']!='train'};assert not reserved&{r['document_hash'] for r in selected};positions=[31,63,95,119];arrays={};tokens=[]
 paths=[Path(__file__),mp,ART/'bilin18_pca_behavior_transfer_bases.pt',SNAPSHOT/'pytorch_model.bin']+[Path(p) for p in sorted({r['source'] for r in selected})];hashes={str(p):sha(p) for p in paths}
 for r in selected:
  if r['source'] not in arrays:arrays[r['source']]=np.load(r['source'],mmap_mode='r')
  ids=np.asarray(arrays[r['source']][r['source_row'],:128],dtype=np.int64);assert hashlib.sha256(ids.tobytes()).hexdigest()==r['window_sha256_int64'];tokens.append(ids.copy())
 tokens=torch.tensor(np.stack(tokens));cfg=dict(experiment='H998',hashes=hashes,documents=selected,positions=positions,split='First192 hash-sorted historical train documents discovery; remaining64 pilot confirmation. No original BSF validation/test document selected. Documents reused, not new corpus or model pretraining holdout.',design='Capture MLP12 writes at four predetermined causal prefix positions per document, plus native next-token diagnostics. Freeze task-trained PCA128; inspect its orthogonal residual. Fit8 spherical residual clusters using discovery only, two seeds,50steps. Save examples for hypothesis generation; no causal or task-discovery claim.',limits='Residual Euclidean geometry and prototype clusters may reflect token/format/topic confounds. Native next tokens are contextual continuations, not supervised task labels. No cluster outcome-based selection.')
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');(ART/f'{S}_manifest.json').write_text(json.dumps(cfg,indent=2)+'\n');mh=sha(ART/f'{S}_manifest.json');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);captures=[];diags=[];first_logits=None
 for st in range(0,256,8):
  ids=tokens[st:st+8].cuda();cache=[]
  h=model.blocks[12].mlp.register_forward_hook(lambda m,a,out:cache.append(out[:,positions].detach().cpu()))
  try:
   x0=F.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None
   for block in model.blocks:x,bus=block(x,bus,x0)
   logits=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,positions]).float()/30)
  finally:h.remove()
  captures.append(cache[0]);lp=logits.double().log_softmax(-1);values,inds=lp.topk(5,dim=-1)
  if st==0:first_logits=logits[0,0].double().cpu()
  for k in range(len(ids)):
   for pi,p in enumerate(positions):
    doc=st+k;prefix=tokens[doc,:p+1].tolist();actual=int(tokens[doc,p+1]);diags.append(dict(document=doc,position=p,split='discovery' if doc<192 else 'confirmation',document_hash=selected[doc]['document_hash'],last_token=int(tokens[doc,p]),actual_next=actual,actual_next_logprob=float(lp[k,pi,actual]),top_ids=inds[k,pi].cpu().tolist(),top_logprobs=values[k,pi].cpu().tolist(),prefix=tok.decode(prefix),last_token_text=tok.decode([prefix[-1]]),actual_next_text=tok.decode([actual]),top_text=[tok.decode([int(i)]) for i in inds[k,pi]]))
  print('CAPTURE',st+8,flush=True)
 X=torch.cat(captures).reshape(-1,1152).double();state=torch.load(paths[2],weights_only=False)['PCA128_expanded6'];B=state['basis'];mu=state['center'];centered=X-mu;z=centered@B;residual=centered-z@B.T;order=z.abs().topk(32,dim=1).indices;support=torch.zeros_like(z).scatter_(1,order,1);scalar_res=centered-(z*support)@B.T;ratios={}
 for name,sl in [('discovery',slice(0,768)),('confirmation',slice(768,None))]:ratios[name]=dict(dense128_error=float(residual[sl].square().sum()/centered[sl].square().sum()),scalar32_error=float(scalar_res[sl].square().sum()/centered[sl].square().sum()),native_next_top1=float(np.mean([r['top_ids'][0]==r['actual_next'] for r in diags[sl]])))
 U=F.normalize(residual.float(),dim=1);clusterings={};cards=[]
 for seed in [9981,9982]:
  gen=torch.Generator().manual_seed(seed);C=U[torch.randperm(768,generator=gen)[:8]].clone()
  for step in range(50):
   lab=(U[:768]@C.T).argmax(1);updated=[]
   for k in range(8):updated.append(F.normalize(U[:768][lab==k].mean(0),dim=0) if bool((lab==k).any()) else C[k])
   C=torch.stack(updated)
  sim=U@C.T;lab=sim.argmax(1);clusterings[seed]=dict(centers=C,labels=lab,similarity=sim)
  for k in range(8):
   ix=torch.where(lab[:768]==k)[0].tolist();ranked=sorted(ix,key=lambda i:float(sim[i,k]),reverse=True);chosen=[];docs=set()
   for i in ranked:
    if diags[i]['document'] not in docs:chosen.append(i);docs.add(diags[i]['document'])
    if len(chosen)==4:break
   counts=Counter(diags[i]['last_token'] for i in ix);cards.append(dict(seed=seed,cluster=k,discovery_states=len(ix),confirmation_states=int((lab[768:]==k).sum()),current_token_mode_fraction=counts.most_common(1)[0][1]/len(ix) if ix else 0,modal_current_token=tok.decode([counts.most_common(1)[0][0]]) if ix else '',exemplars=[dict(index=i,**diags[i]) for i in chosen]))
 y,cap=forward(model,tokens[:1,:32].cuda(),capture=True);logiterror=float((y[0].double().cpu()-first_logits).abs().max());captureerror=float((cap[12][0].double()-X[0]).norm()/X[0].norm());assert logiterror<1e-4 and captureerror<1e-5
 aa=clusterings[9981]['labels'][768:];bb=clusterings[9982]['labels'][768:];upper=torch.triu(torch.ones(256,256,dtype=torch.bool),1);pa=(aa[:,None]==aa[None,:])&upper;pb=(bb[:,None]==bb[None,:])&upper;agreement=float((pa&pb).sum()/(pa|pb).sum())
 audits=dict(hashes_unchanged=all(sha(p)==hashes[str(p)] for p in paths),manifest_unchanged=sha(ART/f'{S}_manifest.json')==mh,original_BSF_nontrain_documents_used=0,pilot_document_splits_disjoint=True,prefix_logits_replay_max_abs=logiterror,prefix_capture_relative_error=captureerror,whole_model_forwards=257,original_reserved_test_evaluations=0,finite=bool(torch.isfinite(X).all()),residual_projection_relative=float((residual@B).norm()/residual.norm()))
 assert audits['hashes_unchanged'] and audits['manifest_unchanged'] and audits['finite'];torch.save(dict(writes=X,tokens=tokens,metadata=diags,residual=residual,clusters=clusterings),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,ratios=ratios,cards=cards,confirmation_cluster_pair_jaccard=agreement,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print('COMPLETE',ratios,agreement,audits,flush=True)
if __name__=='__main__':main()
