"""H639 downstream normalized-attention/MLP-context/MLP-normalization factorial."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_upstream_coefficient_prediction'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_fresh_shift_panel_manifest.json','capture':ART/'bilin18_fresh_shift_panel_captures.pt','source':ART/'bilin18_fresh_shift_panel_captures.pt','internal':ART/'bilin18_fresh_shift_panel_captures.pt','locked':ART/'bilin18_mean_context_shift_locked.pt','discovery':ART/'bilin18_prospective_operator_panel_captures.pt','discovery_manifest':ART/'bilin18_prospective_operator_panel_manifest.json','previous':ART/'bilin18_frozen_shift_transfer_captures.pt','expanded_basis':ART/'bilin18_residual_discovery_expansion_locked.pt','expanded_results':ART/'bilin18_residual_discovery_expansion_captures.pt','late_result':ART/'bilin18_receiver_coefficient_prediction_captures.pt','runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['capture'],weights_only=False);source=torch.load(paths['source'],weights_only=False);internal=torch.load(paths['internal'],weights_only=False);names=list(source['responses']);caches=cap['caches'];bus=cap['bus'];x0=cap['x0'];lookup={(e['family'],e['form'],e['item']):i for i,e in enumerate(rows)};donors={kind:[lookup[e['family'],e['form']^1,e['item']] if kind=='cross' else lookup[e['family'],e['form'],(e['item']+4)%8] for e in rows] for kind in ['cross','within']};cfg=dict(experiment='H641',hashes=hashes,probes=names,axes={'1':'donor bilinear MLP operating context','2':'donor MLP RMS-normalization derivative','4':'donor normalized-attention Jacobian'},design='Full2x2x2 operator factorial over blocks10-17 crossed with native/donor block9 incoming tangents, two donor maps, native/donor readouts. Frozen native states, no fitting. Mask0/mask7 checked against whole-block H638 chain.',limitations='Attention includes its normalization. Joint factor states need not form realizable prompts. Relative gap norms are not causal shares.')
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
 cfg=dict(experiment='H647',hashes=hashes,design='Discovery-only rank1/2/3 SVD bases of concatenated eight-layer context-shift residuals around frozen H641 means. Four original discovery items per family/template pair. Fresh coefficients are oracle Euclidean projections of true donor context residual, shared across probes and layers; evaluate actual composed responses with both endpoint choices.',limitations='Basis fitted on discovery only, coefficients use fresh donor contexts. Euclidean projection is not functionally optimal; this is not an output-space upper bound or predictor. No rank selection.');(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));disc=torch.load(paths['discovery'],weights_only=False);drows=json.loads(paths['discovery_manifest'].read_text());db=defaultdict(list)
 for i,e in enumerate(drows):db[len(e['ids'])].append(i)
 dy={};devals=0
 for l in range(10,18):
  block=model.blocks[l];Y=torch.empty(len(drows),1152,dtype=torch.float64)
  for ii in db.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];x=torch.stack([disc['caches'][l][i] for i in bb]).cuda();e=torch.stack([disc['x0'][i] for i in bb]).cuda();b=torch.stack([disc['bus'][i] for i in bb]).cuda();rawx=block.lambdas[0]*x+block.lambdas[1]*e;att,_=block.attn(F.rms_norm(rawx,(1152,)),b);postx=(rawx[:,-1]+att[:,-1]).double();radius=(postx.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();Y[bb]=(postx/radius).cpu();devals+=len(bb)
  dy[l]=Y
 cfg=dict(experiment='H650',hashes=hashes,design='Rank3 coefficient ridge prediction from receiver raw final-position block9 residual only, before modeled segment. Eight discovery items per family/form; centered linear kernel divided by its discovery mean diagonal. Fixed ridge alpha1; no tuning. Oracle rank3 and mean8 controls; shuffled-label control rotates discovery item labels by4.',limitations='Retrospective evaluation on already-observed fresh panel. Uses receiver block9 state and known template/family, not end-to-end language model. Oracle coefficients evaluation-only. Fixed regularization may underfit.');(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));expanded=torch.load(paths['expanded_basis'],weights_only=False);bases=expanded['bases'];mean8=expanded['means8'];old=torch.load(paths['expanded_results'],weights_only=False)['final_tangents'];late=torch.load(paths['late_result'],weights_only=False)['final_tangents'];dl={(e['family'],e['form'],e['item']):i for i,e in enumerate(drows)};fitmodels={};normalmax=0.;meandiff=0.
 for family in ['copy','fact']:
  for form in range(4):
   B=bases[family,form//2][:3];mu=mean8[family,form//2]*(1 if form%2==0 else -1);ii=[dl[family,form,item] for item in range(8)];di=[dl[family,form^1,item] for item in range(8)];features=disc['native_captures'][9][ii].double();deltas=torch.stack([torch.cat([dy[l][d]-dy[l][i] for l in range(10,18)]) for i,d in zip(ii,di)]);meandiff=max(meandiff,float((deltas.mean(0)-mu).abs().max()));labels=(deltas-mu)@B.T;fc=features.mean(0);xc=features-fc;scale=(xc.square().sum(-1)).mean();K=xc@xc.T/scale;eye=torch.eye(8,dtype=torch.float64);weights={}
   for mode,lab in [('ridge',labels),('shuffle',labels.roll(4,0))]:
    weight=torch.linalg.solve(K+eye,lab);normalmax=max(normalmax,float(((K+eye)@weight-lab).abs().max()));weights[mode]=weight
   fitmodels[family,form]=dict(center=fc,centered_features=xc,scale=scale,weights=weights,labels=labels)
 locked=ART/f'{STEM}_locked.pt';torch.save(dict(models=fitmodels,bases=bases,means8=mean8,alpha=1.),locked);paths['predictor']=locked;hashes['predictor']=hashlib.sha256(locked.read_bytes()).hexdigest();cfg['hashes']=hashes;(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));jj=donors['cross'];shifts={m:{} for m in ['ridge','shuffle']};stacks={m:[] for m in shifts};geometry=[]
 for i,e in enumerate(rows):
  f=fitmodels[e['family'],e['form']];feature=internal['native_captures'][9][i].double();kernel=(feature-f['center'])@f['centered_features'].T/f['scale'];B=bases[e['family'],e['form']//2][:3];mu=mean8[e['family'],e['form']//2]*(1 if e['form']%2==0 else -1)
  # True donor shifts are used below only for evaluation of coefficient error.
  true_delta=torch.cat([factors[l][2][jj[i]].cpu()-factors[l][2][i].cpu() for l in range(10,18)]);truth=(true_delta-mu)@B.T
  for mode in shifts:
   pred=kernel@f['weights'][mode];stacks[mode].append((mu+pred@B).reshape(8,1152));geometry.append(dict(id=e['id'],family=e['family'],form=e['form'],mode=mode,error_squared=float((pred-truth).square().sum()),target_squared=float(truth.square().sum()),predicted_coefficients=pred.tolist(),oracle_coefficients=truth.tolist()))
 for mode in shifts:
  stack=torch.stack(stacks[mode]).cuda()
  for l in range(10,18):shifts[mode][l]=stack[:,l-10]
 X=internal['native_captures'][17].double().cuda();rr=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();W=model.lm_head.weight.double();sat=1-torch.tanh((X/rr)@W.T/30).square();P=source['native_logits'].double().softmax(-1).cuda();records=[];finals={}
 def norm(z):return (((z-(z*P).sum(-1,keepdim=True)).square()*P).sum(-1)).sqrt()
 for name in names:
  finals[name]={}
  for incoming in ['receiver','donor']:
   ff={k:old[name][incoming][k] for k in ['native','exact','mean8_only','mean8_rank3']};ff['late_ridge']=late[name][incoming]['ridge']
   for mode in shifts:
    d=internal['tangents'][name][9].clone()
    if incoming=='donor':d=d[jj]
    for l in range(10,18):
     d=float(model.blocks[l].lambdas[0])*d;d+=attnaction(l,d);x,r,y,L,R,D=factors[l];dd=d.cuda();t=dd/r-x*(dd*x).mean(-1,keepdim=True)/r.pow(3);ys=y+shifts[mode][l];d+=(((ys@L.T)*(t@R.T)+(t@L.T)*(ys@R.T))@D.T).cpu()
    ff[mode]=d
   finals[name][incoming]=ff
   for readout in ['receiver','donor']:
    xx=X if readout=='receiver' else X[jj];rs=rr if readout=='receiver' else rr[jj];ss=sat if readout=='receiver' else sat[jj]
    def read(d):
     dd=d.cuda();return ss*((dd/rs-xx*(dd*xx).mean(-1,keepdim=True)/rs.pow(3))@W.T)
    zz={k:read(v) for k,v in ff.items()};den=norm(zz['exact']-zz['native'])
    for mode,z in zz.items():
     error=norm(z-zz['exact'])/den
     for i,e in enumerate(rows):records.append(dict(id=e['id'],family=e['family'],form=e['form'],probe=name,incoming=incoming,readout=readout,mode=mode,error=float(error[i])))
   print('COMPLETE',name,incoming,flush=True)
 audits=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},new_whole_model_forwards=0,fresh_attention_evaluations=attnevals,discovery_attention_evaluations=devals,batched_jvp_calls=jvps,expanded_mean_replay_max_abs=meandiff,ridge_normal_equation_max_abs=normalmax,finite=all(np.isfinite(r['error']) for r in records));torch.save(dict(final_tangents=finals),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,geometry=geometry,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()
