"""H592: causal linear/quadratic response ablations around native MLP11 output."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_number_mlp11_terms'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_number_mlp11_response_cross.json';saved=torch.load(ART/'bilin18_number_mlp11_response_cross_captures.pt',weights_only=False);ex=saved['examples'];inputs=saved['inputs'];writes=saved['writes'];bp=ART/'bilin18_agreement_write_subspace_locked_bases.pt';b=torch.load(bp,weights_only=False);axis=b['pca'][0]*torch.sign(b['pca'][0]@b['mean']);scale=float(axis@b['mean']);native={}
 for stem in ('bilin18_number_subject_attractor','bilin18_number_fronted_attractor'):native.update(torch.load(ART/f'{stem}_cache.pt',weights_only=False)['native'])
 doses=[.0625,.5];cfg=dict(experiment='H592',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),basis_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),doses=doses,design='MLP11 response replaced by linear term, quadratic term, their sum, or zero, around its same-prompt native output; keep original signed Attention5 perturbation and downstream computation live. All 192 prompts, both doses/signs.',metrics='Reference KL and group-NLL computed in float64 from float32 model logits; margin/group-NLL errors versus sum. Normalize errors by sum-versus-zero-response discrepancy to isolate this MLP contribution rather than whole-network perturbation. Central and per-sign effects separately.',scope='Bilinear expansion after normalization. Incoming normalized-input changes can themselves be nonlinear in injected dose. Linear-only refers to MLP response in that input change, not whole-network linearity.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda');assert not model.config.gated
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 mlp=model.blocks[11].mlp;terms={};errors=[];buckets=defaultdict(list)
 for k,e in ex.items():
  z=inputs[k,0.].cuda();lz=mlp.Left(z);rz=mlp.Right(z)
  for d in doses:
   for sign in (-1,1):
    dose=d*sign;de=(inputs[k,dose]-inputs[k,0.]).cuda();le=mlp.Left(de);re=mlp.Right(de);lin=mlp.Down(lz*re+le*rz);quad=mlp.Down(le*re);terms[k,dose]={'sum':(lin+quad).cpu(),'linear':lin.cpu(),'quadratic':quad.cpu(),'zero':torch.zeros_like(lin).cpu()};actual=(writes[k,dose]-writes[k,0.]).cuda();errors.append(float((lin+quad-actual).norm()/actual.norm()));buckets[len(e['ids'])].append((k,dose))
 result=dict(config=cfg,interventions=[],responses=[],summary={},status='running');outputs={}
 for mode in ('sum','zero','linear','quadratic'):
  for T,jj in buckets.items():
   for st in range(0,len(jj),24):
    bb=jj[st:st+24];don=torch.stack([writes[k,0.]+terms[k,dose][mode] for k,dose in bb]).cuda();delta=torch.stack([axis*scale*dose for k,dose in bb]);
    def h(m,args,out):y=out.clone();y[:,-1]=don.to(y);return y
    handle=mlp.register_forward_hook(h)
    try:zz,_=run(model,torch.tensor([ex[k]['ids'] for k,d in bb],device='cuda'),patch=(5,delta))
    finally:handle.remove()
    for ix,(k,dose) in enumerate(bb):
     z=zz[ix].cpu();outputs[k,dose,mode]=z;ref=outputs[k,dose,'sum'];e=ex[k];gs=e['groups'];lp=z.double().log_softmax(0);rp=ref.double().log_softmax(0);nm=measure(native[k],gs)['margin'];margin=measure(z,gs)['margin'];rm=measure(ref,gs)['margin'];ans=gs[e['subject_number']];result['interventions'].append(dict(id=k,kind=e['kind'],frame=e['frame'],dose=dose,mode=mode,plural_shift=nm-margin,margin_error=margin-rm,reference_kl=float((rp.exp()*(rp-lp)).sum()),group_nll_error=float(torch.logsumexp(rp[ans],0)-torch.logsumexp(lp[ans],0)),response_relative_error=float((terms[k,dose][mode]-terms[k,dose]['sum']).norm()/terms[k,dose]['sum'].norm())))
  print('COMPLETE',mode,flush=True)
 lookup={(c['id'],c['dose'],c['mode']):c for c in result['interventions']}
 for k,e in ex.items():
  for d in doses:
   for mode in ('sum','zero','linear','quadratic'):result['responses'].append(dict(id=k,kind=e['kind'],frame=e['frame'],dose=d,mode=mode,central_plural_response=(lookup[k,d,mode]['plural_shift']-lookup[k,-d,mode]['plural_shift'])/(2*d)))
 for kind in ('regular','irregular'):
  for frame in ('preposition','relative_clause','fronted_preposition'):
   for dose in doses:
    cc=[c for c in result['interventions'] if c['kind']==kind and c['frame']==frame and abs(c['dose'])==dose];zero=[c for c in cc if c['mode']=='zero'];md=np.mean([abs(c['margin_error']) for c in zero]);kd=np.mean([c['reference_kl'] for c in zero])
    for mode in ('sum','zero','linear','quadratic'):
     vv=[c for c in cc if c['mode']==mode];rr=[c['central_plural_response'] for c in result['responses'] if (c['kind'],c['frame'],c['dose'],c['mode'])==(kind,frame,dose,mode)];result['summary'][f'{kind}_{frame}_{dose}_{mode}']=dict(mean_central_response=float(np.mean(rr)),relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/md),relative_reference_kl=float(np.mean([c['reference_kl'] for c in vv])/kd),reference_kl=float(np.mean([c['reference_kl'] for c in vv])),group_nll_abs_error=float(np.mean([abs(c['group_nll_error']) for c in vv])),response_relative_error=float(np.mean([c['response_relative_error'] for c in vv])))
 prior=json.loads(src.read_text());psum={(c['id'],c['dose']):c for c in prior['interventions'] if c['mode']=='recipient_recipient'};old=json.loads((ART/'bilin18_number_response_writers.json').read_text());pzero={(c['id'],c['dose']):c for c in old['interventions'] if c['mode']=='mlp_11'};result['audits']=dict(bilinear_response_max_relative_error=max(errors),sum_replay_max_margin_error=max(abs(c['plural_shift']-psum[c['id'],c['dose']]['plural_shift']) for c in result['interventions'] if c['mode']=='sum'),zero_replay_max_margin_error=max(abs(c['plural_shift']-pzero[c['id'],c['dose']]['plural_shift']) for c in result['interventions'] if c['mode']=='zero'),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['reference_kl','margin_error','group_nll_error']),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),basis_unchanged=cfg['basis_sha256']==hashlib.sha256(bp.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(result['audits'],flush=True)
if __name__=='__main__':main()
