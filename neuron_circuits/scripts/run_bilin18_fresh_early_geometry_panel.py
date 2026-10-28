"""H824: prospective document-disjoint early geometry causal panel, outcome-blind donors."""
import json,hashlib,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_early_bsf_extended import BSF
from run_bilin18_early_local_geometry import sha
S='bilin18_fresh_early_geometry_panel'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();mp=ART/'bilin18_early_bsf_data_manifest.json';old=json.loads(mp.read_text());exclude={r['document_hash'] for r in old['records']};pool={};sources={}
 for name,h in old['source_hashes'].items():
  if not name.endswith('.manifest.json'):continue
  p=Path(name);assert sha(p)==h;j=json.loads(p.read_text());arr=p.with_name(p.name.replace('.manifest.json','.npy'));assert sha(arr)==old['source_hashes'][str(arr)];sources[name]=h;sources[str(arr)]=old['source_hashes'][str(arr)]
  for i,r in enumerate(j['records']):
   dh=r['document_token_sha256']
   if dh not in exclude:pool.setdefault(dh,{'document_hash':dh,'source':str(arr),'source_row':i})
 selected=[pool[h] for h in sorted(pool,key=lambda h:hashlib.sha256(('H824:'+h).encode()).hexdigest())[:1024]];assert len(selected)==1024
 arrays={};tokens=[]
 for r in selected:
  if r['source'] not in arrays:arrays[r['source']]=np.load(r['source'],mmap_mode='r')
  tokens.append(np.asarray(arrays[r['source']][r['source_row'],:128],dtype=np.int64))
 tokens=torch.tensor(np.stack(tokens));positions=list(range(7,128,8));paths={'old_manifest':mp,'prior_geometry':ART/'bilin18_early_local_geometry.json','runner':Path(__file__)}
 for site in ['residual0','residual1']:paths[site]=ART/'early_bsf_extended'/f'{site}_r16_seed681.pt'
 hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H824','hashes':hashes,'documents':1024,'selection':'H824 document hash order from existing H681 source pool, exclude all2560 prior BSF documents including reserved test. No new corpus documents. Novel to this early BSF panel, not all research/model pretraining.','donors':'Only H822 count-eligible token/block clouds. Same token and absolute sampled position, different document, both block active. For each receiver require >=4 eligible donors; nearest and farthest decoded contribution. Require far distance>=1.5*near and near>1e-8.','receiver_selection':'First4 receivers per block by deterministic hash among qualifying candidates. No downstream logits or outcomes read.','future_interventions':'Near/far block directions rescaled to common near distance; controls use training PCA16 and fixed random16 projection of each full donor residual difference rescaled to same norm. Full native-scale donor swaps separate descriptive arm. Near/far is not causal evidence of smooth manifold without downstream comparison.','status':'capture and freeze pair identities, no outcome evaluation'}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');(ART/f'{S}_manifest.json').write_text(json.dumps({'records':selected,'source_hashes':sources,'excluded_document_count':len(exclude)},indent=2)+'\n')
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 out={site:torch.empty(1024,16,1152) for site in ['residual0','residual1']};pidx=torch.tensor(positions,device='cuda')
 for st in range(0,1024,16):
  ids=tokens[st:st+16].cuda();x0=F.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None
  for l in range(2):x,bus=model.blocks[l](x,bus,x0);out[f'residual{l}'][st:st+len(ids)]=x[:,pidx].cpu()
 del model;torch.cuda.empty_cache();oldgeo=json.loads(paths['prior_geometry'].read_text());pairs=[];coverage=[];codes={};reconstruction=[]
 for site in ['residual0','residual1']:
  ck=torch.load(paths[site],weights_only=False);m=BSF(1152,16).cuda();m.load_state_dict(ck['state_dict']);blocks=sorted({r['block'] for r in oldgeo['records'] if r['site']==site});zs=[];ms=[]
  for b in out[site].reshape(-1,1152).split(512):z,mask=m.encode((b.cuda()-ck['mean'].cuda())/ck['scale'].cuda());zs.append(z[:,blocks].cpu());ms.append(mask[:,blocks].cpu())
  z=torch.cat(zs).double();mask=torch.cat(ms).bool();D=m.decoder.detach().cpu().double().reshape(144,16,1152);codes[site]={'blocks':blocks,'z':z,'mask':mask,'scale':ck['scale'],'decoder':D[blocks]}
  for bi,block in enumerate(blocks):
   validtokens={r['token_id'] for r in oldgeo['records'] if r['site']==site and r['block']==block};groups=defaultdict(list)
   for k in mask[:,bi].nonzero().flatten().tolist():
    tid=int(tokens[k//16,positions[k%16]])
    if tid in validtokens:groups[tid,k%16].append(k)
   q,r=torch.linalg.qr(D[block].T,mode='reduced');coords=z[:,bi]@r.T*ck['scale'];candidates=[]
   for (tid,pos),inds in groups.items():
    if len(inds)<5:continue
    xx=coords[inds];dist=torch.cdist(xx,xx);dist.fill_diagonal_(float('inf'));nn=dist.argmin(1);dist.fill_diagonal_(0);ff=dist.argmax(1)
    for a,receiver in enumerate(inds):
     near=inds[int(nn[a])];far=inds[int(ff[a])];dn=float(dist[a,nn[a]]);df=float(dist[a,ff[a]])
     if dn<=1e-8 or df<1.5*dn:continue
     candidates.append({'site':site,'block':block,'receiver':receiver,'near':near,'far':far,'token_id':tid,'position':positions[pos],'near_norm':dn,'far_norm':df,'common_norm':dn,'available_donors':len(inds)-1})
   candidates.sort(key=lambda p:hashlib.sha256(f"H824:{site}:{block}:{p['receiver']}".encode()).hexdigest());keep=candidates[:4];pairs.extend(keep);coverage.append({'site':site,'block':block,'eligible_receivers':len(candidates),'selected_receivers':len(keep)})
   for p in keep:
    for arm in ['near','far']:
     delta=(z[p[arm],bi]-z[p['receiver'],bi])@D[block]*ck['scale'];reconstruction.append(abs(float(delta.norm())-p[arm+'_norm'])/p[arm+'_norm']);assert p['receiver']//16!=p[arm]//16
  print(site,coverage,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'source_hashes_unchanged':all(sha(Path(p))==h for p,h in sources.items()),'documents_disjoint_from_all_old_bsf_splits':not bool(exclude&{r['document_hash'] for r in selected}),'selected_receiver_count':len(pairs),'same_token_position':all(all(int(tokens[p[d]//16,p['position']])==p['token_id'] and positions[p[d]%16]==p['position'] for d in ['receiver','near','far']) for p in pairs),'decoded_distance_max_relative_error':max(reconstruction),'two_block_prefix_forwards':1024,'whole_model_forwards':0,'test_evaluations':0,'finite':all(bool(v.isfinite().all()) for v in out.values())};assert audits['hashes_unchanged'] and audits['source_hashes_unchanged'] and audits['documents_disjoint_from_all_old_bsf_splits'] and audits['same_token_position'] and max(reconstruction)<1e-8
 torch.save({'activations':out,'all_token_ids':tokens,'positions':positions,'codes':codes,'pairs':pairs},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'pairs':pairs,'coverage':coverage,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
