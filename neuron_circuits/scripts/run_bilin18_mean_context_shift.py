"""H639 downstream normalized-attention/MLP-context/MLP-normalization factorial."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_mean_context_shift'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_prospective_operator_panel_manifest.json','capture':ART/'bilin18_prospective_operator_panel_captures.pt','source':ART/'bilin18_prospective_operator_panel_captures.pt','internal':ART/'bilin18_prospective_operator_panel_captures.pt','boundary':ART/'bilin18_prospective_boundary_control_captures.pt','exact':ART/'bilin18_prospective_late_factorial_captures.pt','runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['capture'],weights_only=False);source=torch.load(paths['source'],weights_only=False);internal=torch.load(paths['internal'],weights_only=False);boundary=torch.load(paths['boundary'],weights_only=False)['final_tangents'];names=['original','span2','orthogonal2'];caches=cap['caches'];bus=cap['bus'];x0=cap['x0'];lookup={(e['family'],e['form'],e['item']):i for i,e in enumerate(rows)};donors={kind:[lookup[e['family'],e['form']^1,e['item']] if kind=='cross' else lookup[e['family'],e['form'],(e['item']+4)%8] for e in rows] for kind in ['cross','within']};cfg=dict(experiment='H641',hashes=hashes,probes=names,axes={'1':'donor bilinear MLP operating context','2':'donor MLP RMS-normalization derivative','4':'donor normalized-attention Jacobian'},design='Full2x2x2 operator factorial over blocks10-17 crossed with native/donor block9 incoming tangents, two donor maps, native/donor readouts. Frozen native states, no fitting. Mask0/mask7 checked against whole-block H638 chain.',limitations='Attention includes its normalization. Joint factor states need not form realizable prompts. Relative gap norms are not causal shares.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 buckets=defaultdict(list)
 for i,e in enumerate(rows):buckets[len(e['ids'])].append(i)
 raw={l:{} for l in range(10,18)};post={l:{} for l in range(10,18)};attnevals=0
 for l in range(10,18):
  block=model.blocks[l]
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];x=torch.stack([caches[l][i] for i in bb]).cuda();e=torch.stack([x0[i] for i in bb]).cuda();b=torch.stack([bus[i] for i in bb]).cuda();r=block.lambdas[0]*x+block.lambdas[1]*e;a,_=block.attn(F.rms_norm(r,(1152,)),b);attnevals+=len(bb)
    for k,i in enumerate(bb):raw[l][i]=r[k].cpu();post[l][i]=(r[k,-1]+a[k,-1]).cpu()
 factors={}
 for l in range(10,18):
  X=torch.stack([post[l][i] for i in range(len(rows))]).double().cuda();r=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();m=model.blocks[l].mlp;factors[l]=(X,r,X/r,m.Left.weight.double(),m.Right.weight.double(),m.Down.weight.double())
 jvps=0
 def attnaction(l,d):
  nonlocal jvps
  result=torch.empty_like(d)
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];b=torch.stack([bus[i] for i in bb]).cuda();x=torch.stack([raw[l][i] for i in bb]).cuda();prefix=x[:,:-1];last=x[:,-1];mod=model.blocks[l].attn
    def fn(tail):return mod(F.rms_norm(torch.cat([prefix,tail[:,None]],1),(1152,)),b)[0][:,-1]
    _,out=torch.func.jvp(fn,(last,),(d[bb].cuda().float(),));result[bb]=out.cpu().double();jvps+=1
  return result
 # Fit only first four item pairs; reverse shifts are exact negatives.
 exact=torch.load(paths['exact'],weights_only=False)['final_tangents'];means={};geometry=[]
 for l,(_,_,y,*_) in factors.items():
  means[l]={}
  for family in ['copy','fact']:
   for form in range(4):
    ii=[i for i,e in enumerate(rows) if e['family']==family and e['form']==form and e['item']<4];jj=[donors['cross'][i] for i in ii];means[l][family,form]=(y[jj]-y[ii]).mean(0)
  for i,e in enumerate(rows):
   delta=y[donors['cross'][i]]-y[i];mu=means[l][e['family'],e['form']];geometry.append(dict(id=e['id'],family=e['family'],form=e['form'],item=e['item'],layer=l,relative_error=float((delta-mu).norm()/delta.norm()),cosine=float((delta@mu)/(delta.norm()*mu.norm()))))
 locked=ART/f'{STEM}_locked.pt';torch.save({l:{k:v.cpu() for k,v in mm.items()} for l,mm in means.items()},locked);paths['locked']=locked;hashes['locked']=hashlib.sha256(locked.read_bytes()).hexdigest();cfg=dict(experiment='H641',hashes=hashes,fit_items=[0,1,2,3],heldout_items=[4,5,6,7],design='Mean normalized MLP-context shift per family/template/layer fit on first four pairs only. Predict donor y as receiver y + mean shift; no renormalization. Compare exact context-only replacement, zero shift, and norm-matched shift from other family at same template index. Native attention/MLP normalization, both input/readout choices.',limitations='Retrospective item holdout after H640; no fitting to output derivatives, three probes all excluded from shift fitting. Donor input/readout are still accessed in those explicitly conditioned modes. Not an end-to-end donor prediction or low-rank operator proof.');(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2))
 X=internal['native_captures'][17].double().cuda();rr=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();W=model.lm_head.weight.double();sat=1-torch.tanh((X/rr)@W.T/30).square();P=source['native_logits'].double().softmax(-1).cuda();records=[];finals={};replays=[];jj=donors['cross']
 def center(z):return z-(z*P).sum(-1,keepdim=True)
 def norms(z):return (center(z).square()*P).sum(-1).sqrt()
 for name in names:
  finals[name]={}
  for incoming in ['receiver','donor']:
   ff={'native':exact[name][f'cross_{incoming}_0'],'exact':exact[name][f'cross_{incoming}_1']}
   for mode in ['predicted','other_family']+(['replay_native','replay_exact'] if name=='original' else []):
    d=internal['tangents'][name][9].clone()
    if incoming=='donor':d=d[jj]
    for l in range(10,18):
     d=float(model.blocks[l].lambdas[0])*d;d+=attnaction(l,d);x,r,y,L,R,D=factors[l];dd=d.cuda();t=dd/r-x*(x*dd).mean(-1,keepdim=True)/r.pow(3)
     if mode=='replay_native':ys=y
     elif mode=='replay_exact':ys=y[jj]
     else:
      shifts=[]
      for e in rows:
       mu=means[l][e['family'],e['form']]
       if mode=='other_family':
        other=means[l]['fact' if e['family']=='copy' else 'copy',e['form']];mu=other*(mu.norm()/other.norm())
       shifts.append(mu)
      ys=y+torch.stack(shifts)
     d+=(((ys@L.T)*(t@R.T)+(t@L.T)*(ys@R.T))@D.T).cpu()
    if mode.startswith('replay_'):replays.append(float(((d-ff[mode.removeprefix('replay_')]).norm(dim=-1)/ff[mode.removeprefix('replay_')].norm(dim=-1)).max()))
    else:ff[mode]=d
   finals[name][incoming]=ff
   for readout in ['receiver','donor']:
    xx=X if readout=='receiver' else X[jj];rs=rr if readout=='receiver' else rr[jj];ss=sat if readout=='receiver' else sat[jj];zz={}
    for mode,d in ff.items():
     dd=d.cuda();zz[mode]=ss*((dd/rs-xx*(xx*dd).mean(-1,keepdim=True)/rs.pow(3))@W.T)
    fullgap=norms(source['responses'][name][jj].cuda()-source['responses'][name].cuda());effect=norms(zz['exact']-zz['native']);exactnorm=norms(zz['exact'])
    for mode,z in zz.items():
     predictionerror=norms(z-zz['exact']);remain=norms(source['responses'][name][jj].cuda()-z)/fullgap
     for i,e in enumerate(rows):records.append(dict(id=e['id'],family=e['family'],form=e['form'],item=e['item'],split='fit' if e['item']<4 else 'heldout',probe=name,incoming=incoming,readout=readout,mode=mode,error_over_exact_effect=float(predictionerror[i]/effect[i]),error_over_exact_response=float(predictionerror[i]/exactnorm[i]),remaining_gap=float(remain[i])))
   print('COMPLETE',name,incoming,flush=True)
 audits=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},new_whole_model_forwards=0,cached_attention_evaluations=attnevals,batched_jvp_calls=jvps,exact_native_replay_max_error=max(replays),fit_contexts=32,heldout_contexts=32,finite=all(np.isfinite(r['error_over_exact_effect']) for r in records));torch.save(dict(final_tangents=finals),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,geometry=geometry,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()
