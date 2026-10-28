"""H639 downstream normalized-attention/MLP-context/MLP-normalization factorial."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_shift_interaction_orders'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_fresh_shift_panel_manifest.json','capture':ART/'bilin18_fresh_shift_panel_captures.pt','source':ART/'bilin18_fresh_shift_panel_captures.pt','internal':ART/'bilin18_fresh_shift_panel_captures.pt','locked':ART/'bilin18_mean_context_shift_locked.pt','ablation':ART/'bilin18_shift_layer_ablation_captures.pt','target':ART/'bilin18_frozen_shift_transfer_captures.pt','runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['capture'],weights_only=False);source=torch.load(paths['source'],weights_only=False);internal=torch.load(paths['internal'],weights_only=False);names=list(source['responses']);caches=cap['caches'];bus=cap['bus'];x0=cap['x0'];lookup={(e['family'],e['form'],e['item']):i for i,e in enumerate(rows)};donors={kind:[lookup[e['family'],e['form']^1,e['item']] if kind=='cross' else lookup[e['family'],e['form'],(e['item']+4)%8] for e in rows] for kind in ['cross','within']};cfg=dict(experiment='H641',hashes=hashes,probes=names,axes={'1':'donor bilinear MLP operating context','2':'donor MLP RMS-normalization derivative','4':'donor normalized-attention Jacobian'},design='Full2x2x2 operator factorial over blocks10-17 crossed with native/donor block9 incoming tangents, two donor maps, native/donor readouts. Frozen native states, no fitting. Mask0/mask7 checked against whole-block H638 chain.',limitations='Attention includes its normalization. Joint factor states need not form realizable prompts. Relative gap norms are not causal shares.')
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
 cfg=dict(experiment='H644',hashes=hashes,design='Exact degree<=8 polynomial in shared amplitude s of frozen context shifts, all four probes, both incoming/readout choices. Fixed truncation orders0-8; coefficients recursively propagated through native and shift maps. Controls: constant/native, first-order/additive, full/full-predictor, direct s=.5 and-1 original receiver-input.',limitations='Interaction order counts distinct shifted layers in a local tangent chain, not finite activation Taylor order. Sum of all pairs/triples, not selected circuits. No fitting or subset selection.');(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));ablation=torch.load(paths['ablation'],weights_only=False)['final_tangents'];target=torch.load(paths['target'],weights_only=False)['final_tangents'];shifts={l:torch.stack([means[l][e['family'],e['form']] for e in rows]) for l in range(10,18)};jj=donors['cross'];allcoeff={};valid=[];amplitude_valid=[]
 def mlpaction(l,pre,shift=False):
  x,r,y,L,R,D=factors[l];dd=pre.cuda();t=dd/r-x*(x*dd).mean(-1,keepdim=True)/r.pow(3);ys=shifts[l] if shift else y;return (((ys@L.T)*(t@R.T)+(t@L.T)*(ys@R.T))@D.T).cpu()
 def before_mlp(l,d):
  d=float(model.blocks[l].lambdas[0])*d;return d+attnaction(l,d)
 for name in names:
  allcoeff[name]={}
  for incoming in ['receiver','donor']:
   d=internal['tangents'][name][9].clone()
   if incoming=='donor':d=d[jj]
   coeff=[d]
   for l in range(10,18):
    pre=[before_mlp(l,c) for c in coeff];nxt=[c+mlpaction(l,c) for c in pre]+[torch.zeros_like(d)]
    for k,c in enumerate(pre):nxt[k+1]+=mlpaction(l,c,True)
    coeff=nxt
   allcoeff[name][incoming]=coeff
   for label,actual,expected in [('constant',coeff[0],ablation[name][incoming][0]),('first_order',coeff[0]+coeff[1],ablation[name][incoming]['additive']),('full',sum(coeff),ablation[name][incoming][255])]:valid.append(dict(probe=name,incoming=incoming,control=label,max_relative_error=float(((actual-expected).norm(dim=-1)/expected.norm(dim=-1)).max())))
   if name=='original' and incoming=='receiver':
    for amp in [.5,-1.]:
     dd=internal['tangents'][name][9].clone()
     for l in range(10,18):
      pre=before_mlp(l,dd);dd=pre+mlpaction(l,pre)+amp*mlpaction(l,pre,True)
     pred=sum(amp**k*c for k,c in enumerate(coeff));amplitude_valid.append(dict(amplitude=amp,error=float(((dd-pred).norm(dim=-1)/dd.norm(dim=-1)).max())))
   print('COEFFICIENTS',name,incoming,flush=True)
 X=internal['native_captures'][17].double().cuda();rr=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();W=model.lm_head.weight.double();sat=1-torch.tanh((X/rr)@W.T/30).square();P=source['native_logits'].double().softmax(-1).cuda();records=[];order_norms=[]
 def center(z):return z-(z*P).sum(-1,keepdim=True)
 def norms(z):return (center(z).square()*P).sum(-1).sqrt()
 for name in names:
  for incoming in ['receiver','donor']:
   coeff=allcoeff[name][incoming]
   for readout in ['receiver','donor']:
    xx=X if readout=='receiver' else X[jj];rs=rr if readout=='receiver' else rr[jj];ss=sat if readout=='receiver' else sat[jj]
    def read(d):
     dd=d.cuda();return ss*((dd/rs-xx*(xx*dd).mean(-1,keepdim=True)/rs.pow(3))@W.T)
    zn=read(coeff[0]);zf=read(sum(coeff));ze=read(target[name][incoming]['exact']);den=norms(zf-zn);exactden=norms(ze-zn);acc=torch.zeros_like(zn)
    for k,c in enumerate(coeff):
     cz=read(c);acc+=cz;er=norms(acc-zf)/den;ee=norms(acc-ze)/exactden;cn=norms(cz)/den
     for i,e in enumerate(rows):
      records.append(dict(id=e['id'],family=e['family'],form=e['form'],probe=name,incoming=incoming,readout=readout,order=k,error_over_full_effect=float(er[i]),error_over_exact_effect=float(ee[i])))
      if k:order_norms.append(dict(id=e['id'],family=e['family'],form=e['form'],probe=name,incoming=incoming,readout=readout,order=k,norm_over_full_effect=float(cn[i])))
 audits=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},new_whole_model_forwards=0,cached_attention_evaluations=attnevals,batched_jvp_calls=jvps,coefficient_recovery_max_error=max(v['max_relative_error'] for v in valid),amplitude_recovery_max_error=max(v['error'] for v in amplitude_valid),finite=all(np.isfinite(r['error_over_full_effect']) for r in records));torch.save(dict(coefficients=allcoeff),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,order_norms=order_norms,validation=valid,amplitude_validation=amplitude_valid,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()
