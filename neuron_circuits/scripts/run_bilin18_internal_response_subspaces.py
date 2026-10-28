"""H635 internal response coverage, exact residual skip control, native readout."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_internal_response_subspaces'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();mp=ART/'bilin18_cross_task_response_panel_manifest.json';cp=ART/'bilin18_cross_task_response_panel_captures.pt';pp=ART/'bilin18_emission_probe_directions_locked.pt';cfgp=ROOT/'configs/bilin18_cross_task_response_subspaces.json';rows=json.loads(mp.read_text());source=torch.load(cp,weights_only=False);probes=torch.load(pp,weights_only=False);oldcfg=json.loads(cfgp.read_text());names=list(probes);fit=[i for i,e in enumerate(rows) if e['split']=='fit'];train_names=oldcfg['fit_directions'];stages=[5,9,13,17];ranks=[4,16,32,100];paths={'manifest':mp,'source':cp,'probes':pp,'split_config':cfgp,'runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()}
 cfg=dict(experiment='H635',hashes=hashes,stages=stages,ranks=ranks,fit_directions=train_names,design='Capture internal residual central tangents at blocks5/9/13/17, +/-0.02. Fit discovery-only Euclidean SVD of full tangent or innovation after subtracting exact residual identity path. Reconstruct with oracle Euclidean coefficients, restoring identity path for innovation. Map final reconstructions through exact receiver-native readout. Global20 contexts x5 probes, same retrospective holdout as H634.',limitations='Oracle coefficients, fixed input directions; identity-path subtraction is algebraic and leaves all attention/MLP effects in innovation. Euclidean residual projection is not output-optimal. Local derivatives, not full circuit replacement.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 buckets=defaultdict(list)
 for i,e in enumerate(rows):buckets[len(e['ids'])].append(i)
 forwards=0
 def evaluate(q=None,alpha=0.):
  nonlocal forwards
  out=torch.empty_like(source['native_logits']);caps={l:torch.empty(len(rows),1152) for l in stages}
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];hs=[]
    if q is not None:
     def edit(m,args,value):y,b=value;y=y.clone();y[:,-1]+=alpha*q.to(y);return y,b
     hs.append(model.blocks[5].attn.register_forward_hook(edit))
    for l in stages:
     def capture(m,args,value,l=l):caps[l][bb]=value[0][:,-1].cpu()
     hs.append(model.blocks[l].register_forward_hook(capture))
    try:z,_=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'))
    finally:
     for h in hs:h.remove()
    out[bb]=z.cpu();forwards+=len(bb)
  return out,caps
 native,nativecaps=evaluate();tangents={};replay=0.
 for name,q in probes.items():
  zm,cm=evaluate(q,-.02);zp,cpp=evaluate(q,.02);tangents[name]={l:(cpp[l].double()-cm[l].double())/.04 for l in stages};replay=max(replay,float((((zp.double()-zm.double())/.04)-source['responses'][name]).abs().max()));print('CAPTURED',name,flush=True)
 scales={};scale=1.
 for l in range(5,18):
  if l>5:scale*=float(model.blocks[l].lambdas[0])
  if l in stages:scales[l]=scale
 bases={};records=[];output_records=[];validation=[];maxorth=0.
 for l in stages:
  allg=torch.stack([tangents[n][l] for n in names]).cuda();skip=torch.stack([probes[n]*scales[l] for n in names]).cuda()[:,None,:].expand_as(allg)
  for kind in ['full','innovation']:
   target=allg if kind=='full' else allg-skip;M=torch.stack([target[names.index(n),i] for i in fit for n in train_names]);_,sv,B=torch.linalg.svd(M,full_matrices=False);bases[f'{l}_{kind}']=B.cpu();maxorth=max(maxorth,float((B@B.T-torch.eye(len(B),device='cuda')).abs().max()))
   for rank in ranks:
    basis=B[:rank];recon=(target@basis.T)@basis;total=recon if kind=='full' else recon+skip;err=(target-recon).norm(dim=-1)/target.norm(dim=-1);fullerr=(allg-total).norm(dim=-1)/allg.norm(dim=-1)
    for n,name in enumerate(names):
     for i,e in enumerate(rows):records.append(dict(id=e['id'],family=e['family'],form=e['form'],split=e['split'],probe=name,direction_split='fit' if name in train_names else 'heldout',stage=l,kind=kind,rank=rank,target_relative_error=float(err[n,i]),full_relative_error=float(fullerr[n,i]),innovation_to_full_norm=float((allg[n,i]-skip[n,i]).norm()/allg[n,i].norm())))
    if l==17:
     W=model.lm_head.weight.double();X=nativecaps[17].double().cuda();rr=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
     for i,e in enumerate(rows):
      p=native[i].double().cuda().softmax(0);x=X[i];r=rr[i];sat=1-torch.tanh((x/r)@W.T/30).square()
      def readout(d):return sat*((d/r-x*(x*d).mean(-1,keepdim=True)/r.pow(3))@W.T)
      predicted=readout(total[:,i]);truth=torch.stack([source['responses'][n][i] for n in names]).cuda()
      def center(d):return d-(d*p).sum(-1,keepdim=True)
      truecenter=center(truth);errors=((center(predicted-truth).square()*p).sum(-1)/(truecenter.square()*p).sum(-1)).sqrt()
      if kind=='full' and rank==4:
       exact=readout(allg[:,i]);er=((center(exact-truth).square()*p).sum(-1)/(truecenter.square()*p).sum(-1)).sqrt();validation.extend([dict(id=e['id'],probe=n,error=float(er[k])) for k,n in enumerate(names)])
      for k,n in enumerate(names):output_records.append(dict(id=e['id'],family=e['family'],form=e['form'],split=e['split'],probe=n,direction_split='fit' if n in train_names else 'heldout',kind=kind,rank=rank,relative_error=float(errors[k])))
   print('FITTED',l,kind,flush=True)
 audits=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},native_forwards=forwards,native_replay_max_abs=float((native-source['native_logits']).abs().max()),derivative_replay_max_abs=replay,basis_orthogonality_max=maxorth,readout_validation_max=max(r['error'] for r in validation),finite=all(np.isfinite(r['relative_error']) for r in output_records))
 torch.save(dict(native_captures=nativecaps,tangents=tangents,skip_scales=scales,bases=bases),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,output_records=output_records,validation=validation,skip_scales=scales,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()
