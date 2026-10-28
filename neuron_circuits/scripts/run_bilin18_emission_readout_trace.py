"""H625 downstream response trace and analytic final-readout controls."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_emission_readout_trace'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();mp=ART/'bilin18_emission_surface_transfer_manifest.json';vp=ART/'bilin18_emission_surface_transfer_locked.pt';prior=ART/'bilin18_emission_logit_response_captures.pt';rows=json.loads(mp.read_text());v=torch.load(vp,weights_only=False)['common'];old=torch.load(prior,weights_only=False);paths={'manifest':mp,'vector':vp,'prior':prior,'runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};cfg=dict(experiment='H625',hashes=hashes,design='Central .02 response capture after blocks5-17 and their attention/MLP writes. Analytic derivative of final float32-epsilon RMS, linear unembedding, tanh cap. First4-item capital/pool/cell mean pre-readout response through each native readout; own response through mean capital readout. Same fit split asH624, no new direction fit.',scope='Local derivative diagnostics; mean pre-readout response uses internal observed responses, not prediction from text. Trace divergence is not standalone causal attribution to a layer.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 buckets=defaultdict(list)
 for i,e in enumerate(rows):buckets[len(e['ids'])].append(i)
 captures={};zs={};stages=[f'{name}{l}' for l in range(5,18) for name in ['block','attn','mlp']]
 for alpha in [-.02,0.,.02]:
  captures[alpha]={s:torch.empty(len(rows),1152) for s in stages};zs[alpha]=torch.empty_like(old['logits'][0.])
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];cap={};hs=[]
    def edit(m,args,out):y,b=out;y=y.clone();y[:,-1]+=alpha*v.to(y);return y,b
    hs.append(model.blocks[5].attn.register_forward_hook(edit))
    for l in range(5,18):
     for name,mod in [('block',model.blocks[l]),('attn',model.blocks[l].attn),('mlp',model.blocks[l].mlp)]:
      def hook(m,args,out,name=name,l=l):cap[f'{name}{l}']=(out if name=='mlp' else out[0])[:,-1].cpu()
      hs.append(mod.register_forward_hook(hook))
    try:z,_=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'))
    finally:
     for h in hs:h.remove()
    zs[alpha][bb]=z.cpu()
    for s in stages:captures[alpha][s][bb]=cap[s]
  print('CAPTURED',alpha,flush=True)
 D={s:(captures[.02][s].double()-captures[-.02][s].double())/.04 for s in stages};train=[i for i,e in enumerate(rows) if e['item']<4];captrain=[i for i in train if rows[i]['family']=='capital' and rows[i]['form']==0];cells=sorted({(e['family'],e['form']) for e in rows});X=captures[0.]['block17'].double();DX=D['block17'];W=model.lm_head.weight.double();eps=torch.finfo(torch.float32).eps
 def readout(dx,x):
  dx=dx.cuda();x=x.cuda();r=(x.square().mean(-1,keepdim=True)+eps).sqrt();y=x/r;dy=dx/r-x*(x*dx).mean(-1,keepdim=True)/r.pow(3);raw=y@W.T;return ((1-torch.tanh(raw/30).square())*(dy@W.T)).cpu()
 raw=(X.cuda()/(X.cuda().square().mean(-1,keepdim=True)+eps).sqrt())@W.T;replay=float((30*torch.tanh(raw/30).cpu()-zs[0.].double()).abs().max());actual=(zs[.02].double()-zs[-.02].double())/.04;pred={'local_native':readout(DX,X),'capital_native':readout(DX[captrain].mean(0).expand_as(DX),X),'pooled_native':readout(DX[train].mean(0).expand_as(DX),X)};cellmean=torch.empty_like(DX)
 for family,form in cells:
  ii=[i for i,e in enumerate(rows) if e['family']==family and e['form']==form];tt=[i for i in ii if i in train];cellmean[ii]=DX[tt].mean(0)
 pred['cell_native']=readout(cellmean,X);pred['local_shared_readout']=sum(readout(DX,X[i].expand_as(X)) for i in captrain)/len(captrain);pred['capital_shared_readout']=sum(readout(DX[captrain].mean(0).expand_as(DX),X[i].expand_as(X)) for i in captrain)/len(captrain)
 records=[];trace=[];chain=[]
 for i,e in enumerate(rows):
  p=zs[0.][i].double().softmax(0)
  def center(x):return x-(x*p).sum()
  def norm(x):return (center(x).square()*p).sum().sqrt()
  g=actual[i];split='fit' if i in train else 'holdout'
  for name,response in pred.items():records.append(dict(id=e['id'],family=e['family'],form=e['form'],split=split,mode=name,response_error=float(norm(response[i]-g)/norm(g)),cosine=float((center(response[i])*center(g)*p).sum()/(norm(response[i])*norm(g)))))
  for l in range(5,18):
   dx=D[f'block{l}'][i];mean=D[f'block{l}'][captrain].mean(0);gain=float((dx@mean)/mean.square().sum());trace.append(dict(id=e['id'],family=e['family'],form=e['form'],split=split,layer=l,norm=float(dx.norm()),capital_mean_error=float((dx-mean).norm()/dx.norm()),capital_cosine=float(torch.nn.functional.cosine_similarity(dx,mean,dim=0)),oracle_gain=gain,scalar_error=float((dx-gain*mean).norm()/dx.norm()),mlp_response_norm=float(D[f'mlp{l}'][i].norm()),attention_response_norm=float(D[f'attn{l}'][i].norm())))
   incoming=v.double() if l==5 else float(model.blocks[l].lambdas[0])*D[f'block{l-1}'][i]+D[f'attn{l}'][i]
   chain.append(float((dx-incoming-D[f'mlp{l}'][i]).norm()/dx.norm()))
 result=dict(config=cfg,records=records,trace=trace,summary={},trace_summary={},status='complete')
 for family,form in cells:
  for split in ['fit','holdout']:
   for name in pred:
    rr=[r for r in records if r['family']==family and r['form']==form and r['split']==split and r['mode']==name];result['summary'][f'{family}_{form}_{split}_{name}']={n:float(np.mean([r[n] for r in rr])) for n in ['response_error','cosine']}
   for l in range(5,18):
    rr=[r for r in trace if r['family']==family and r['form']==form and r['split']==split and r['layer']==l];result['trace_summary'][f'{family}_{form}_{split}_{l}']={n:float(np.mean([r[n] for r in rr])) for n in ['norm','capital_mean_error','capital_cosine','scalar_error','mlp_response_norm','attention_response_norm']}
 result['audits']=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},evaluations=len(rows)*3,readout_native_max_logit_error=replay,prior_native_max_logit_error=float((zs[0.]-old['logits'][0.]).abs().max()),chain_rule_max_relative_error=max(chain),input_response_max_relative_error=float(((D['attn5']-v.double()).norm(dim=-1)/v.norm()).max()),analytic_response_max_relative_error=max(r['response_error'] for r in records if r['mode']=='local_native'),finite=all(np.isfinite(r['response_error']) for r in records));result['elapsed_seconds']=time.time()-start;torch.save(dict(captures=captures,tangents=D,logits=zs,predicted_readout=pred),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
