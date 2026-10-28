"""H825 frozen fresh donor direction interventions."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
from run_bilin18_early_local_geometry import sha
S='bilin18_fresh_geometry_causal'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'panel':ART/'bilin18_fresh_early_geometry_panel_captures.pt','panel_json':ART/'bilin18_fresh_early_geometry_panel.json','training':ART/'bilin18_early_bsf_data_captures.pt','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['panel'],weights_only=False);train=torch.load(paths['training'],weights_only=False);pairs=data['pairs'];bases={};patches={};normerr=[]
 cfg={'experiment':'H825','hashes':hashes,'design':'All56 frozen H824 receivers,112 near/far relationships. Common-norm block/PCA16/random16 directions; separately native-scale block swaps. Zero edit per receiver. Native final-position distribution references.','selection':'No outcome based changes; training PCA from original2048 documents only; random seed825+layer.','metrics':'KL(receiver||edited); donor KL gain KL(donor||receiver)-KL(donor||edited); total variation; same receiver near/far effect differences. Different donor targets preclude interpreting raw near/far gain as direct superiority.','limitations':'Small dependent block/receiver sample; natural donor does not ensure partial transplant on manifold; shared norm shrinks far edits. No task correctness claim.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 for site in ['residual0','residual1']:
  x=train['activations'][site][:2048].reshape(-1,1152).double();x-=x.mean(0);_,v=torch.linalg.eigh(x.T@x/len(x));gen=torch.Generator().manual_seed(825+int(site[-1]));bases[site]={'pca':v[:,-16:],'random':torch.linalg.qr(torch.randn(1152,16,generator=gen,dtype=torch.float64)).Q}
 for i,p in enumerate(pairs):
  c=data['codes'][p['site']];bi=c['blocks'].index(p['block']);z=c['z'];D=c['decoder'][bi];raw=data['activations'][p['site']].reshape(-1,1152).double()
  for arm in ['near','far']:
   don=p[arm];block=(z[don,bi]-z[p['receiver'],bi])@D*c['scale'];delta=raw[don]-raw[p['receiver']]
   for mode in ['block','pca','random','native_block']:
    v=block if mode in ['block','native_block'] else (delta@bases[p['site']][mode])@bases[p['site']][mode].T
    assert v.norm()>1e-12
    if mode!='native_block':v=v*(p['common_norm']/v.norm());normerr.append(abs(float(v.norm())/p['common_norm']-1))
    patches[i,arm,mode]=v
  patches[i,'zero','zero']=torch.zeros(1152,dtype=torch.float64)
 del train;model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 endpoints=sorted({p[k] for p in pairs for k in ['receiver','near','far']});native={};replay=[];forwards=0;buckets=defaultdict(list)
 for ep in endpoints:buckets[data['positions'][ep%16]].append(ep)
 for pos,ee in buckets.items():
  for st in range(0,len(ee),8):
   batch=ee[st:st+8];capt={};hs=[]
   for layer in [0,1]:
    def get(m,args,out,layer=layer):capt[layer]=out[0][:,-1].cpu()
    hs.append(model.blocks[layer].register_forward_hook(get))
   try:z,_=run(model,torch.stack([data['all_token_ids'][ep//16,:pos+1] for ep in batch]).cuda())
   finally:
    for h in hs:h.remove()
   for n,ep in enumerate(batch):
    native[ep]=z[n].cpu()
    for layer in [0,1]:truth=data['activations'][f'residual{layer}'][ep//16,ep%16];replay.append(float((capt[layer][n]-truth).norm()/truth.norm()))
   forwards+=len(batch)
 records=[];outputs={};zeroerr=[];jobs=defaultdict(list)
 for key in patches:
  i,arm,mode=key;p=pairs[i];jobs[p['site'],p['position']].append(key)
 for (site,pos),keys in jobs.items():
  for st in range(0,len(keys),8):
   batch=keys[st:st+8];patch=torch.stack([patches[k] for k in batch]);layer=int(site[-1])
   def edit(m,args,out):
    x,bus=out;x=x.clone();x[:,-1]+=patch.to(x);return x,bus
   h=model.blocks[layer].register_forward_hook(edit)
   try:z,_=run(model,torch.stack([data['all_token_ids'][pairs[k[0]]['receiver']//16,:pos+1] for k in batch]).cuda())
   finally:h.remove()
   forwards+=len(batch)
   for n,key in enumerate(batch):
    i,arm,mode=key;p=pairs[i];out=z[n].cpu();outputs[key]=out;base=native[p['receiver']].double().log_softmax(0);ed=out.double().log_softmax(0)
    if mode=='zero':zeroerr.append(float((out-native[p['receiver']]).abs().max()));continue
    don=native[p[arm]].double().log_softmax(0);dk=float((don.exp()*(don-base)).sum());gain=float((don.exp()*(ed-base)).sum());records.append({'pair':i,'site':site,'block':p['block'],'receiver':p['receiver'],'receiver_document':p['receiver']//16,'donor':p[arm],'arm':arm,'mode':mode,'patch_norm':float(patches[key].norm()),'donor_kl_baseline':dk,'donor_kl_gain':gain,'receiver_kl':float((base.exp()*(base-ed)).sum()),'total_variation':float((base.exp()-ed.exp()).abs().sum()/2)})
  print('DONE',site,pos,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'prefix_replay_max_relative':max(replay),'zero_edit_logits_max_abs':max(zeroerr),'common_norm_max_relative':max(normerr),'native_endpoints':len(endpoints),'receivers':len(pairs),'intervention_forwards':len(patches),'whole_model_forwards':forwards,'finite':all(all(torch.isfinite(torch.tensor(r[k])).item() for k in ['receiver_kl','donor_kl_gain','total_variation']) for r in records),'test_evaluations':0};print(audits,flush=True);assert audits['hashes_unchanged'] and audits['finite'] and max(replay)<1e-4 and max(zeroerr)<1e-4 and max(normerr)<1e-10
 torch.save({'outputs':outputs,'native':native,'patches':patches,'bases':bases,'pairs':pairs},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
if __name__=='__main__':main()
