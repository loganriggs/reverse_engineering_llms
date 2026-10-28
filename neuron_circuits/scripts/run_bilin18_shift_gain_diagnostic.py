"""H639 downstream normalized-attention/MLP-context/MLP-normalization factorial."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_shift_gain_diagnostic'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_fresh_shift_panel_manifest.json','capture':ART/'bilin18_fresh_shift_panel_captures.pt','source':ART/'bilin18_fresh_shift_panel_captures.pt','internal':ART/'bilin18_fresh_shift_panel_captures.pt','locked':ART/'bilin18_mean_context_shift_locked.pt','previous':ART/'bilin18_frozen_shift_transfer_captures.pt','runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['capture'],weights_only=False);source=torch.load(paths['source'],weights_only=False);internal=torch.load(paths['internal'],weights_only=False);names=list(source['responses']);caches=cap['caches'];bus=cap['bus'];x0=cap['x0'];lookup={(e['family'],e['form'],e['item']):i for i,e in enumerate(rows)};donors={kind:[lookup[e['family'],e['form']^1,e['item']] if kind=='cross' else lookup[e['family'],e['form'],(e['item']+4)%8] for e in rows] for kind in ['cross','within']};cfg=dict(experiment='H641',hashes=hashes,probes=names,axes={'1':'donor bilinear MLP operating context','2':'donor MLP RMS-normalization derivative','4':'donor normalized-attention Jacobian'},design='Full2x2x2 operator factorial over blocks10-17 crossed with native/donor block9 incoming tangents, two donor maps, native/donor readouts. Frozen native states, no fitting. Mask0/mask7 checked against whole-block H638 chain.',limitations='Attention includes its normalization. Joint factor states need not form realizable prompts. Relative gap norms are not causal shares.')
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
 means={l:{k:v.cuda() for k,v in mm.items()} for l,mm in torch.load(paths['locked'],weights_only=False).items()}
 cfg=dict(experiment='H642',hashes=hashes,design='Frozen H641 mean shifts, no refit, all64 fresh prompts held out. Original anchor plus3 seeded independent fresh probes orthogonal to prior9. Native/exact/predicted/other-family contexts, both input/readout conditions.',limitations='Fixed prior templates; new identities and probes, not new tasks or instructions. Conditioned donor input/readout still use donor information. Kenya/Nairobi replaced by Belgium/Brussels before inference because Nairobi is multi-token.');(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2))
 cfg=dict(experiment='H645',hashes=hashes,design='Oracle geometric scalar gains: project true donor-minus-receiver context shifts onto frozen mean shifts per layer or with one shared gain across8 layers. Propagate to compare exact context intervention. Also analytically fit a scalar to the full predicted output effect. All oracle diagnostics use test-target information; no predictive fit.',limitations='Geometric gains optimize context Euclidean error, not final response. Output-effect scalar is not an optimum over8 layer gains. No claim that all functional gain models are exhausted.');(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));previous=torch.load(paths['previous'],weights_only=False)['final_tangents'];jj=donors['cross'];shifts={l:torch.stack([means[l][e['family'],e['form']] for e in rows]) for l in range(10,18)};deltas={l:factors[l][2][jj]-factors[l][2] for l in range(10,18)};gains={l:(deltas[l]*shifts[l]).sum(-1)/shifts[l].square().sum(-1) for l in range(10,18)};globalgain=sum((deltas[l]*shifts[l]).sum(-1) for l in range(10,18))/sum(shifts[l].square().sum(-1) for l in range(10,18));geometry=[]
 for l in range(10,18):
  for i,e in enumerate(rows):
   delta=deltas[l][i];mu=shifts[l][i];geometry.append(dict(id=e['id'],family=e['family'],form=e['form'],layer=l,layer_gain=float(gains[l][i]),chain_gain=float(globalgain[i]),mean_error=float((delta-mu).norm()/delta.norm()),layer_gain_error=float((delta-gains[l][i]*mu).norm()/delta.norm()),chain_gain_error=float((delta-globalgain[i]*mu).norm()/delta.norm())))
 X=internal['native_captures'][17].double().cuda();rr=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();W=model.lm_head.weight.double();sat=1-torch.tanh((X/rr)@W.T/30).square();P=source['native_logits'].double().softmax(-1).cuda();records=[];finals={};replays=[]
 def center(z):return z-(z*P).sum(-1,keepdim=True)
 def norms(z):return (center(z).square()*P).sum(-1).sqrt()
 for name in names:
  finals[name]={}
  for incoming in ['receiver','donor']:
   ff={k:previous[name][incoming][k] for k in ['native','exact','predicted']}
   for mode in ['layer_gain','chain_gain']+(['replay'] if name=='original' else []):
    d=internal['tangents'][name][9].clone()
    if incoming=='donor':d=d[jj]
    for l in range(10,18):
     d=float(model.blocks[l].lambdas[0])*d;d+=attnaction(l,d);x,r,y,L,R,D=factors[l];dd=d.cuda();t=dd/r-x*(x*dd).mean(-1,keepdim=True)/r.pow(3);gain=gains[l] if mode=='layer_gain' else globalgain if mode=='chain_gain' else torch.ones_like(globalgain);ys=y+gain[:,None]*shifts[l];d+=(((ys@L.T)*(t@R.T)+(t@L.T)*(ys@R.T))@D.T).cpu()
    if mode=='replay':replays.append(float(((d-ff['predicted']).norm(dim=-1)/ff['predicted'].norm(dim=-1)).max()))
    else:ff[mode]=d
   finals[name][incoming]=ff
   for readout in ['receiver','donor']:
    xx=X if readout=='receiver' else X[jj];rs=rr if readout=='receiver' else rr[jj];ss=sat if readout=='receiver' else sat[jj]
    def read(d):
     dd=d.cuda();return ss*((dd/rs-xx*(xx*dd).mean(-1,keepdim=True)/rs.pow(3))@W.T)
    zz={k:read(v) for k,v in ff.items()};targeteffect=center(zz['exact']-zz['native']);predicted_effect=center(zz['predicted']-zz['native']);sg=(P*targeteffect*predicted_effect).sum(-1)/(P*predicted_effect.square()).sum(-1);zz['output_gain']=zz['native']+sg[:,None]*predicted_effect;den=norms(targeteffect)
    for mode,z in zz.items():
     er=norms(z-zz['exact'])/den
     for i,e in enumerate(rows):records.append(dict(id=e['id'],family=e['family'],form=e['form'],probe=name,incoming=incoming,readout=readout,mode=mode,error_over_exact_effect=float(er[i]),output_gain=float(sg[i]) if mode=='output_gain' else None))
   print('COMPLETE',name,incoming,flush=True)
 audits=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},new_whole_model_forwards=0,cached_attention_evaluations=attnevals,batched_jvp_calls=jvps,predictor_replay_max_error=max(replays),finite=all(np.isfinite(r['error_over_exact_effect']) for r in records));torch.save(dict(final_tangents=finals,layer_gains={l:v.cpu() for l,v in gains.items()},chain_gain=globalgain.cpu()),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,geometry=geometry,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()
