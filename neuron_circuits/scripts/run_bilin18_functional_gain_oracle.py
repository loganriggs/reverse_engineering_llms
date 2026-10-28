"""H639 downstream normalized-attention/MLP-context/MLP-normalization factorial."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_functional_gain_oracle'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_fresh_shift_panel_manifest.json','capture':ART/'bilin18_fresh_shift_panel_captures.pt','source':ART/'bilin18_fresh_shift_panel_captures.pt','internal':ART/'bilin18_fresh_shift_panel_captures.pt','locked':ART/'bilin18_mean_context_shift_locked.pt','ablation':ART/'bilin18_shift_layer_ablation_captures.pt','previous':ART/'bilin18_frozen_shift_transfer_captures.pt','runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['capture'],weights_only=False);source=torch.load(paths['source'],weights_only=False);internal=torch.load(paths['internal'],weights_only=False);names=list(source['responses']);caches=cap['caches'];bus=cap['bus'];x0=cap['x0'];lookup={(e['family'],e['form'],e['item']):i for i,e in enumerate(rows)};donors={kind:[lookup[e['family'],e['form']^1,e['item']] if kind=='cross' else lookup[e['family'],e['form'],(e['item']+4)%8] for e in rows] for kind in ['cross','within']};cfg=dict(experiment='H641',hashes=hashes,probes=names,axes={'1':'donor bilinear MLP operating context','2':'donor MLP RMS-normalization derivative','4':'donor normalized-attention Jacobian'},design='Full2x2x2 operator factorial over blocks10-17 crossed with native/donor block9 incoming tangents, two donor maps, native/donor readouts. Frozen native states, no fitting. Mask0/mask7 checked against whole-block H638 chain.',limitations='Attention includes its normalization. Joint factor states need not form realizable prompts. Relative gap norms are not causal shares.')
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
 cfg=dict(experiment='H646',hashes=hashes,fit_probes=['original','fresh0','fresh1'],heldout_probe='fresh2',design='Full-minus-leave-one tangent differences give8 gain Jacobian columns at gain1. Fit per-context oracle correction to exact-context output target with relative-effect weighted least squares pooled over3 probes, separate receiver/receiver and donor/donor endpoints. Evaluate withheld fresh2 and actual composed full/half steps. No gain clipping, no adaptive step selection.',limitations='Uses target responses per test prompt; not a predictive model. One local Gauss-Newton step, not global optimum of gain family. Linearized residual is only local.');(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));old=torch.load(paths['previous'],weights_only=False)['final_tangents'];ab=torch.load(paths['ablation'],weights_only=False)['final_tangents'];jj=donors['cross'];shifts={l:torch.stack([means[l][e['family'],e['form']] for e in rows]) for l in range(10,18)};X=internal['native_captures'][17].double().cuda();rr=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();W=model.lm_head.weight.double();sat=1-torch.tanh((X/rr)@W.T/30).square();P=source['native_logits'].double().softmax(-1).cuda();gains={};fit_audit=[];linear={};normeq=0.
 def readone(d,i):
  dd=d.cuda();x=X[i];r=rr[i];return sat[i]*((dd/r-x*(dd*x).mean(-1,keepdim=True)/r.pow(3))@W.T)
 for endpoint in ['receiver','donor']:
  correction=[]
  for i,e in enumerate(rows):
   ri=i if endpoint=='receiver' else jj[i];p=P[i];gram=torch.zeros(8,8,device='cuda',dtype=torch.float64);rhs=torch.zeros(8,device='cuda',dtype=torch.float64)
   def center(z):return z-(z*p).sum(-1,keepdim=True)
   for name in cfg['fit_probes']:
    full=old[name][endpoint]['predicted'][i];J=torch.stack([full-ab[name][endpoint][255^(1<<k)][i] for k in range(8)]);features=center(readone(J,ri));res=center(readone(old[name][endpoint]['exact'][i]-full,ri));effect=center(readone(old[name][endpoint]['exact'][i]-old[name][endpoint]['native'][i],ri));weight=p/(effect.square()*p).sum();gram+=(features*weight)@features.T;rhs+=(features*weight)@res
   step=torch.linalg.pinv(gram,hermitian=True,rtol=1e-10)@rhs;correction.append(step.cpu());normeq=max(normeq,float((gram@step-rhs).norm()/(rhs.norm()+1e-30)));ev=torch.linalg.eigvalsh(gram);fit_audit.append(dict(id=e['id'],endpoint=endpoint,condition=float(ev.max()/ev.min()),gain_min=float((1+step).min()),gain_max=float((1+step).max()),step_norm=float(step.norm())))
  gains[endpoint]=torch.stack(correction)
  print('FIT',endpoint,flush=True)
 records=[];finals={}
 for name in names:
  finals[name]={}
  for endpoint in ['receiver','donor']:
   correction=gains[endpoint];ff={'native':old[name][endpoint]['native'],'predicted':old[name][endpoint]['predicted'],'exact':old[name][endpoint]['exact']};J=torch.stack([ff['predicted']-ab[name][endpoint][255^(1<<k)] for k in range(8)],dim=1);ff['linear_full']=ff['predicted']+(J*correction[:,:,None]).sum(1);ff['linear_half']=ff['predicted']+.5*(J*correction[:,:,None]).sum(1)
   for step_size in [1.,.5]:
    d=internal['tangents'][name][9].clone()
    if endpoint=='donor':d=d[jj]
    for l in range(10,18):
     d=float(model.blocks[l].lambdas[0])*d;d+=attnaction(l,d);x,r,y,L,R,D=factors[l];dd=d.cuda();t=dd/r-x*(x*dd).mean(-1,keepdim=True)/r.pow(3);scale=(1+step_size*correction[:,l-10]).cuda();ys=y+scale[:,None]*shifts[l];d+=(((ys@L.T)*(t@R.T)+(t@L.T)*(ys@R.T))@D.T).cpu()
    ff['actual_full' if step_size==1 else 'actual_half']=d
   finals[name][endpoint]=ff;indices=list(range(len(rows))) if endpoint=='receiver' else jj;xx=X[indices];rs=rr[indices];ss=sat[indices]
   def read(d):
    dd=d.cuda();return ss*((dd/rs-xx*(dd*xx).mean(-1,keepdim=True)/rs.pow(3))@W.T)
   def norm(z):return (((z-(z*P).sum(-1,keepdim=True)).square()*P).sum(-1)).sqrt()
   zz={k:read(d) for k,d in ff.items()};den=norm(zz['exact']-zz['native'])
   for mode,z in zz.items():
    err=norm(z-zz['exact'])/den
    for i,e in enumerate(rows):records.append(dict(id=e['id'],family=e['family'],form=e['form'],probe=name,probe_split='fit' if name in cfg['fit_probes'] else 'heldout',endpoint=endpoint,mode=mode,error=float(err[i])))
   print('PROPAGATED',name,endpoint,flush=True)
 audits=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},new_whole_model_forwards=0,cached_attention_evaluations=attnevals,batched_jvp_calls=jvps,normal_equation_relative_max=normeq,finite=all(np.isfinite(r['error']) for r in records));torch.save(dict(corrections=gains,final_tangents=finals),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,fit_audit=fit_audit,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()
