"""H681: document-separated early activation panel for task-blind BSF pilots."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_early_bsf_data'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();pool={};sources={}
 for p in sorted(ART.glob('*documents.manifest.json')):
  j=json.loads(p.read_text());rs=j.get('records',[]);arr=p.with_name(p.name.replace('.manifest.json','.npy'))
  if not rs or not arr.exists() or not all('document_token_sha256' in v for v in rs):continue
  x=np.load(arr,mmap_mode='r')
  if x.ndim!=2 or len(x)!=len(rs) or x.shape[1]<128:continue
  sources[str(p)]=sha(p);sources[str(arr)]=sha(arr)
  for i,r in enumerate(rs):pool.setdefault(r['document_token_sha256'],dict(document_hash=r['document_token_sha256'],source=str(arr),source_row=i))
 assert len(pool)>=2560
 ordered=sorted(pool,key=lambda h:hashlib.sha256(('H681:'+h).encode()).hexdigest());selected=[pool[h] for h in ordered[:2560]];arrays={};tokens=[]
 for r in selected:
  if r['source'] not in arrays:arrays[r['source']]=np.load(r['source'],mmap_mode='r')
  tokens.append(np.asarray(arrays[r['source']][r['source_row'],:128],dtype=np.int64))
 tokens=np.stack(tokens);assert tokens.min()>=0 and tokens.max()<50257
 for i,r in enumerate(selected):r['split']='train' if i<2048 else ('validation' if i<2304 else 'test');r['window_sha256_int64']=hashlib.sha256(tokens[i].tobytes()).hexdigest()
 manifest=dict(selection_seed='H681 hash-sort',unique_pool_documents=len(pool),source_hashes=sources,records=selected);mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(manifest,indent=2));paths={'manifest':mp,'runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};positions=list(range(7,128,8));cfg=dict(experiment='H681',hashes=hashes,design='Reuse existing document-hashed FineWeb windows; deduplicate documents across63panels then deterministic hash-sort.2048train/256validation/256test documents,16fixed positions per128token window. Sites: persistent layer0 values and residual after blocks0,1,2. No task labels/prompts used.',training_plan='Pilot block dimensions1,4,16 at fixed total decoder width2304 and active scalar budget32; three seeds. Train centered/global-scaled activations, no per-example normalization. Choose checkpoints only on validation; test reserved for final evaluation. Compare reconstruction, effective block rank, occupancy and across-seed subspace stability before interpreting blocks.',limitations='Documents reused from research, not model-training holdout. Natural corpus duplicates beyond exact full-document hashes may remain. First128 tokens of existing sampled windows, not necessarily document beginnings. Purely lexical persistent values require identity-conditioned controls.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 sites=['value0','residual0','residual1','residual2'];out={s:torch.empty(2560,16,1152) for s in sites};pidx=torch.tensor(positions,device='cuda')
 for st in range(0,2560,16):
  ids=torch.tensor(tokens[st:st+16],device='cuda');x0=F.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None
  for l in range(3):
   x,bus=model.blocks[l](x,bus,x0);out[f'residual{l}'][st:st+len(ids)]=x[:,pidx].cpu()
   if l==0:out['value0'][st:st+len(ids)]=bus.reshape(len(ids),128,1152)[:,pidx].cpu()
  if st%512==0:print('COLLECT',st,flush=True)
 split_hashes={s:{r['document_hash'] for r in selected if r['split']==s} for s in ['train','validation','test']};assert not(split_hashes['train']&split_hashes['validation'] or split_hashes['train']&split_hashes['test'] or split_hashes['validation']&split_hashes['test']);summary={s:{'rms':float(v.square().mean().sqrt()),'finite':bool(torch.isfinite(v).all())} for s,v in out.items()}
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},source_hashes_unchanged=all(sha(Path(p))==h for p,h in sources.items()),document_splits_disjoint=True,documents=2560,new_documents=0,whole_model_forwards=0,three_block_prefix_forwards=2560,finite=all(x['finite'] for x in summary.values()));assert audits['source_hashes_unchanged'] and audits['finite'];torch.save(dict(activations=out,token_ids=torch.tensor(tokens[:,positions]),all_token_ids=torch.tensor(tokens),positions=positions,splits={'train':[0,2048],'validation':[2048,2304],'test':[2304,2560]}),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,summary=summary,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits)
if __name__=='__main__':main()
