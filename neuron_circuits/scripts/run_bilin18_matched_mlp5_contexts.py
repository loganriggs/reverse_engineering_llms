"""H636 matched construction/item factor contrasts at the first affected MLP."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_matched_mlp5_contexts'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();mp=ART/'bilin18_cross_task_response_panel_manifest.json';pp=ART/'bilin18_emission_probe_directions_locked.pt';sp=ART/'bilin18_cross_task_response_panel_captures.pt';ip=ART/'bilin18_internal_response_subspaces_captures.pt';rows=json.loads(mp.read_text());probes=torch.load(pp,weights_only=False);source=torch.load(sp,weights_only=False);internal=torch.load(ip,weights_only=False);paths={'manifest':mp,'probes':pp,'source':sp,'internal':ip,'runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};propnames=['original','span2','orthogonal2'];cfg=dict(experiment='H636',hashes=hashes,propagation_probes=propnames,design='All9 probes local MLP5 factor contrasts. Donors: opposite template same item, or same template item+4 modulo8 (preserves agreement number). Counterfactual context, normalization, or both. Propagate original/span2/orthogonal2 through native downstream Jacobians. No fitting or outcome selection.',limitations='Within-template controls change item; they are not a perfect causal minimal pair. Hybrid factors need not occur naturally. Local derivatives only; no accuracy intervention.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 buckets=defaultdict(list)
 for i,e in enumerate(rows):buckets[len(e['ids'])].append(i)
 raw=torch.empty(len(rows),1152);caches={l:{} for l in range(6,18)};bus={};x0={};native=torch.empty_like(source['native_logits'])
 for ii in buckets.values():
  for st in range(0,len(ii),16):
   bb=ii[st:st+16];hs=[];cap={}
   def pre5(m,args):cap['mixed']=m.lambdas[0]*args[0]+m.lambdas[1]*args[2]
   def attn5(m,args,out):raw[bb]=(cap['mixed'][:,-1]+out[0][:,-1]).cpu()
   hs.append(model.blocks[5].register_forward_pre_hook(pre5));hs.append(model.blocks[5].attn.register_forward_hook(attn5))
   for l in range(6,18):
    def pre(m,args,l=l):
     for k,i in enumerate(bb):
      caches[l][i]=args[0][k].cpu().clone()
      if l==6:bus[i]=args[1][k].cpu().clone();x0[i]=args[2][k].cpu().clone()
    hs.append(model.blocks[l].register_forward_pre_hook(pre))
   try:z,_=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'));native[bb]=z.cpu()
   finally:
    for h in hs:h.remove()
 lookup={(e['family'],e['form'],e['item']):i for i,e in enumerate(rows)};donors={kind:[lookup[e['family'],1-e['form'],e['item']] if kind=='cross' else lookup[e['family'],e['form'],(e['item']+4)%8] for e in rows] for kind in ['cross','within']};X=raw.double().cuda();r=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();y=X/r;m=model.blocks[5].mlp;L=m.Left.weight.double();R=m.Right.weight.double();D=m.Down.weight.double()
 def bilin(y,t):return ((y@L.T)*(t@R.T)+(t@L.T)*(y@R.T))@D.T
 local=[];initials={};validation=[];decompmax=0.
 for name,qcpu in probes.items():
  q=qcpu.cuda();t=q/r-X*(X*q).mean(-1,keepdim=True)/r.pow(3);g=bilin(y,t);initials[name]={'native':(q+g).cpu()};measured=internal['tangents'][name][5]-qcpu;validation.append(dict(probe=name,max_relative_error=float(((g.cpu()-measured).norm(dim=-1)/measured.norm(dim=-1)).max())))
  for kind,jj in donors.items():
   hcontext=bilin(y[jj],t);hnorm=bilin(y,t[jj]);hboth=bilin(y[jj],t[jj]);interaction=bilin(y[jj]-y,t[jj]-t);decompmax=max(decompmax,float(((hboth-g)-(hcontext-g)-(hnorm-g)-interaction).abs().max()))
   for mode,h in [('context',hcontext),('norm',hnorm),('both',hboth)]:
    initials[name][f'{kind}_{mode}']=(q+h).cpu()
    for i,e in enumerate(rows):local.append(dict(id=e['id'],family=e['family'],form=e['form'],item=e['item'],probe=name,donor_kind=kind,donor_id=rows[jj[i]]['id'],mode=mode,change_norm=float((h[i]-g[i]).norm()),relative_change=float((h[i]-g[i]).norm()/g[i].norm()),native_norm=float(g[i].norm()),interaction_norm=float(interaction[i].norm())))
  print('LOCAL',name,flush=True)
 finalX=internal['native_captures'][17].double().cuda();rr=(finalX.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();W=model.lm_head.weight.double();sat=1-torch.tanh((finalX/rr)@W.T/30).square();outputs={};outrecords=[];jvps=0;chainerrs=[]
 for name in propnames:
  outputs[name]={}
  for mode,startd in initials[name].items():
   d=startd
   for l in range(6,18):
    result=torch.empty_like(d)
    for ii in buckets.values():
     for st in range(0,len(ii),16):
      bb=ii[st:st+16];x=torch.stack([caches[l][i] for i in bb]).cuda();b=torch.stack([bus[i] for i in bb]).cuda();e0=torch.stack([x0[i] for i in bb]).cuda();prefix=x[:,:-1];last=x[:,-1];block=model.blocks[l]
      def fn(tail):return block(torch.cat([prefix,tail[:,None]],1),b,e0)[0][:,-1]
      _,v=torch.func.jvp(fn,(last,),(d[bb].cuda().float(),));result[bb]=v.cpu().double();jvps+=1
    d=result
   d=d.cuda();dz=sat*((d/rr-finalX*(finalX*d).mean(-1,keepdim=True)/rr.pow(3))@W.T);outputs[name][mode]=dz.cpu()
  for i,e in enumerate(rows):
   p=native[i].double().softmax(0)
   def center(v):return v-(v*p).sum()
   def norm(v):return (center(v).square()*p).sum().sqrt()
   g=outputs[name]['native'][i];chainerrs.append(float(norm(g-source['responses'][name][i])/norm(source['responses'][name][i])))
   for mode,vv in outputs[name].items():
    h=vv[i];outrecords.append(dict(id=e['id'],family=e['family'],form=e['form'],item=e['item'],probe=name,mode=mode,native_norm=float(norm(g)),change_norm=float(norm(h-g)),relative_change=float(norm(h-g)/norm(g)),target_derivative=float(center(h)[e['answer_id']])))
  print('PROPAGATED',name,flush=True)
 audits=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},native_forwards=len(rows),native_replay_max_abs=float((native-source['native_logits']).abs().max()),local_derivative_max_error=max(v['max_relative_error'] for v in validation),factor_decomposition_max_abs=decompmax,native_chain_max_error=max(chainerrs),batched_jvp_calls=jvps,finite=all(np.isfinite(v['relative_change']) for v in outrecords));torch.save(dict(raw_mlp5=raw,initials=initials,outputs=outputs,caches=caches,bus=bus,x0=x0),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,local_records=local,output_records=outrecords,validation=validation,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()
