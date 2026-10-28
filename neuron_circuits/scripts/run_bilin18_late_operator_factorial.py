"""H639 downstream normalized-attention/MLP-context/MLP-normalization factorial."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_late_operator_factorial'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_cross_task_response_panel_manifest.json','capture':ART/'bilin18_matched_mlp5_contexts_captures.pt','source':ART/'bilin18_cross_task_response_panel_captures.pt','internal':ART/'bilin18_internal_response_subspaces_captures.pt','boundary':ART/'bilin18_boundary9_factorial_captures.pt','runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['capture'],weights_only=False);source=torch.load(paths['source'],weights_only=False);internal=torch.load(paths['internal'],weights_only=False);boundary=torch.load(paths['boundary'],weights_only=False)['final_tangents'];names=['original','span2','orthogonal2'];caches=cap['caches'];bus=cap['bus'];x0=cap['x0'];lookup={(e['family'],e['form'],e['item']):i for i,e in enumerate(rows)};donors={kind:[lookup[e['family'],1-e['form'],e['item']] if kind=='cross' else lookup[e['family'],e['form'],(e['item']+4)%8] for e in rows] for kind in ['cross','within']};cfg=dict(experiment='H639',hashes=hashes,probes=names,axes={'1':'donor bilinear MLP operating context','2':'donor MLP RMS-normalization derivative','4':'donor normalized-attention Jacobian'},design='Full2x2x2 operator factorial over blocks10-17 crossed with native/donor block9 incoming tangents, two donor maps, native/donor readouts. Frozen native states, no fitting. Mask0/mask7 checked against whole-block H638 chain.',limitations='Attention includes its normalization. Joint factor states need not form realizable prompts. Relative gap norms are not causal shares.')
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
 X=internal['native_captures'][17].double().cuda();rr=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();W=model.lm_head.weight.double();sat=1-torch.tanh((X/rr)@W.T/30).square();P=source['native_logits'].double().softmax(-1).cuda();records=[];finals={};validation=[]
 def centered(z):return z-(z*P).sum(-1,keepdim=True)
 def norms(z):return (centered(z).square()*P).sum(-1).sqrt()
 for name in names:
  finals[name]={}
  nativeZ=source['responses'][name].cuda()
  for kind,jj in donors.items():
   donorZ=nativeZ[jj];gap=norms(donorZ-nativeZ)
   for incoming in ['receiver','donor']:
    for mask in range(8):
     d=internal['tangents'][name][9].clone()
     if incoming=='donor':d=d[jj]
     for l in range(10,18):
      d=float(model.blocks[l].lambdas[0])*d;ad=attnaction(l,d[jj])[jj] if mask&4 else attnaction(l,d);d+=ad;x,r,y,L,R,D=factors[l];xx=x[jj] if mask&2 else x;rs=r[jj] if mask&2 else r;ys=y[jj] if mask&1 else y;dd=d.cuda();t=dd/rs-xx*(xx*dd).mean(-1,keepdim=True)/rs.pow(3);d+=(((ys@L.T)*(t@R.T)+(t@L.T)*(ys@R.T))@D.T).cpu()
     key=f'{kind}_{incoming}_{mask}';finals[name][key]=d
     if mask in [0,7]:
      if mask==0:target=boundary[name]['native' if incoming=='receiver' else kind]
      else:target=boundary[name]['native' if incoming=='donor' else kind][jj]
      validation.append(dict(probe=name,key=key,error=float(((d-target).norm(dim=-1)/target.norm(dim=-1)).max())))
     dd=d.cuda()
     for readout in ['receiver','donor']:
      xx=X if readout=='receiver' else X[jj];rs=rr if readout=='receiver' else rr[jj];ss=sat if readout=='receiver' else sat[jj];z=ss*((dd/rs-xx*(xx*dd).mean(-1,keepdim=True)/rs.pow(3))@W.T);remain=norms(donorZ-z)/gap;change=norms(z-nativeZ)/gap
      for i,e in enumerate(rows):records.append(dict(id=e['id'],family=e['family'],form=e['form'],probe=name,kind=kind,incoming=incoming,mask=mask,readout=readout,remaining_gap=float(remain[i]),change_over_gap=float(change[i])))
    print('COMPLETE',name,kind,incoming,flush=True)
 audits=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},new_whole_model_forwards=0,cached_attention_evaluations=attnevals,batched_jvp_calls=jvps,whole_block_replay_max_error=max(r['error'] for r in validation),finite=all(np.isfinite(v['remaining_gap']) for v in records));torch.save(dict(final_tangents=finals),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,validation=validation,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()
