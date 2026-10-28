"""H594: full-discovery-span diagnostic for context and incoming-response coverage."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_number_bilinear_span'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_number_bilinear_subspace.json';saved=torch.load(ART/'bilin18_number_mlp11_response_cross_captures.pt',weights_only=False);all_ex=saved['examples'];inputs=saved['inputs'];writes=saved['writes'];ex={k:e for k,e in all_ex.items() if e['frame']!='preposition'};idx={(e['kind'],e['item'],e['subject_number'],e['attractor_number'],e['frame']):k for k,e in all_ex.items()};pk={k:idx[e['kind'],e['item'],e['subject_number'],e['attractor_number'],'preposition'] for k,e in ex.items()};bp=ART/'bilin18_agreement_write_subspace_locked_bases.pt';b=torch.load(bp,weights_only=False);axis=b['pca'][0]*torch.sign(b['pca'][0]@b['mean']);scale=float(axis@b['mean']);doses=[.0625,.5];ranks=[32]
 cfg=dict(experiment='H594',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),number_basis_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),doses=doses,ranks=ranks,fit='Reuse all 32 H593 discovery basis directions without refitting; test full available context span, incoming-response span, and both. No outcomes used to alter basis.',identity='Context-switch response T(dc,e)=Down(Left(dc)*Right(e)+Left(e)*Right(dc)). Quadratic-in-e term cancels exactly.',design='Project context only, incoming response only, or both onto the entire stored 32-direction discovery spans. Full-context-switch and unchanged-context references. Both signs/doses and all H593 cells retained.',scope='Actual context/input coefficients remain required. Irregular nouns and relative-clause rows are held out from basis fitting but previously explored. No novel corpus or learned coefficient generator. Rank1-both is one scalar-times-scalar output vector, not yet a semantic unit.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));priorbases=torch.load(ART/'bilin18_number_bilinear_subspace_locked_bases.pt',weights_only=False);train=priorbases['training_ids'];A=priorbases['context'];B=priorbases['incoming'];cs=priorbases['context_singular_values'];es=priorbases['incoming_singular_values'];random={};basispath=ART/f'{STEM}_locked_bases.pt';torch.save(dict(context=A,incoming=B,context_singular_values=cs,incoming_singular_values=es,random=random,training_ids=train),basispath);bh=hashlib.sha256(basispath.read_bytes()).hexdigest();print('LOCKED BASES',bh,flush=True)
 model=load_full_model('cuda');assert not model.config.gated
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 mlp=model.blocks[11].mlp
 def T(c,e):
  c=c.cuda();e=e.cuda();return mlp.Down(mlp.Left(c)*mlp.Right(e)+mlp.Left(e)*mlp.Right(c)).cpu()
 vecs={};buckets=defaultdict(list);identityerrors=[];normerrors=[];modes=['full','native']+[f'{which}_{r}' for r in ranks for which in ['context','incoming','both']]
 for k,e in ex.items():
  dc=inputs[pk[k],0.]-inputs[k,0.]
  for dose in [sgn*d for d in doses for sgn in (-1,1)]:
   de=inputs[k,dose]-inputs[k,0.];full=T(dc,de);vv={'full':full,'native':torch.zeros_like(full)}
   for r in ranks:
    pc=A[:r].T@(A[:r]@dc);pe=B[:r].T@(B[:r]@de);vv[f'context_{r}']=T(pc,de);vv[f'incoming_{r}']=T(dc,pe);vv[f'both_{r}']=T(pc,pe)
   cn=(A[:1]@dc).norm();en=(B[:1]@de).norm()
   for s,(ra,rb) in random.items():
    rc=ra.T@(ra@dc);re=rb.T@(rb@de);rc=rc*cn/rc.norm().clamp_min(1e-20);re=re*en/re.norm().clamp_min(1e-20);normerrors.extend([float(abs(rc.norm()-cn)/cn.clamp_min(1e-20)),float(abs(re.norm()-en)/en.clamp_min(1e-20))]);vv[f'random{s}_both1']=T(rc,re)
   actual=saved['hybrids'][k,dose,'preposition_recipient']-saved['hybrids'][k,dose,'recipient_recipient'];identityerrors.append(float((actual-full).norm()/full.norm()))
   for mode,v in vv.items():vecs[k,dose,mode]=v
   buckets[len(e['ids'])].append((k,dose))
 print('LOCKED',len(vecs),'interventions',flush=True);refs={};result=dict(config=cfg,basis_sha256=bh,discovery_energy={str(r):dict(context=float(cs[:r].square().sum()/cs.square().sum()),incoming=float(es[:r].square().sum()/es.square().sum())) for r in ranks},interventions=[],summary={},status='running')
 for mode in modes:
  for length,jj in buckets.items():
   for st in range(0,len(jj),24):
    bb=jj[st:st+24];don=torch.stack([writes[k,0.]+saved['hybrids'][k,dose,'recipient_recipient']+vecs[k,dose,mode] for k,dose in bb]).cuda()
    def h(m,args,out):out=out.clone();out[:,-1]=don.to(out);return out
    handle=mlp.register_forward_hook(h)
    try:zz,_=run(model,torch.tensor([ex[k]['ids'] for k,d in bb],device='cuda'),patch=(5,torch.stack([axis*scale*dose for k,dose in bb])))
    finally:handle.remove()
    for ix,(k,dose) in enumerate(bb):
     z=zz[ix].cpu();e=ex[k];gs=e['groups'];ans=gs[e['subject_number']]
     if mode in ('full','native'):refs[k,dose,mode]=z
     ref=refs[k,dose,'full'];lp=z.double().log_softmax(0);rp=ref.double().log_softmax(0);margin=measure(z,gs)['margin'];rm=measure(ref,gs)['margin'];f=vecs[k,dose,'full'];result['interventions'].append(dict(id=k,kind=e['kind'],frame=e['frame'],dose=dose,mode=mode,margin=margin,margin_error=margin-rm,reference_kl=float((rp.exp()*(rp-lp)).sum()),group_nll_error=float(torch.logsumexp(rp[ans],0)-torch.logsumexp(lp[ans],0)),internal_response_error=float((vecs[k,dose,mode]-f).norm()/f.norm())))
  print('COMPLETE',mode,flush=True)
 for kind in ('regular','irregular'):
  for frame in ('fronted_preposition','relative_clause'):
   for dose in doses:
    cc=[c for c in result['interventions'] if c['kind']==kind and c['frame']==frame and abs(c['dose'])==dose];base=[c for c in cc if c['mode']=='native'];md=np.mean([abs(c['margin_error']) for c in base]);kd=np.mean([c['reference_kl'] for c in base])
    for mode in modes:
     vv=[c for c in cc if c['mode']==mode];result['summary'][f'{kind}_{frame}_{dose}_{mode}']=dict(internal_response_error=float(np.mean([c['internal_response_error'] for c in vv])),relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/md),relative_reference_kl=float(np.mean([c['reference_kl'] for c in vv])/kd),reference_kl=float(np.mean([c['reference_kl'] for c in vv])),group_nll_abs_error=float(np.mean([abs(c['group_nll_error']) for c in vv])))
 prior=json.loads((ART/'bilin18_number_mlp11_response_cross.json').read_text());old={(c['id'],c['dose'],c['mode']):c for c in prior['interventions']};nat={}
 for stem in ('bilin18_number_subject_attractor','bilin18_number_fronted_attractor'):nat.update(torch.load(ART/f'{stem}_cache.pt',weights_only=False)['native'])
 replay=[]
 for c in result['interventions']:
  if c['mode'] not in ('full','native'):continue
  nm=measure(nat[c['id']],ex[c['id']]['groups'])['margin'];om='preposition_recipient' if c['mode']=='full' else 'recipient_recipient';replay.append(abs(nm-c['margin']-old[c['id'],c['dose'],om]['plural_shift']))
 result['audits']=dict(context_switch_identity_max_relative_error=max(identityerrors),matched_input_norm_max_relative_error=max(normerrors,default=0.),reference_replay_max_margin_error=max(replay),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ('margin_error','reference_kl','internal_response_error')),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),basis_unchanged=bh==hashlib.sha256(basispath.read_bytes()).hexdigest(),number_basis_unchanged=cfg['number_basis_sha256']==hashlib.sha256(bp.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(result['audits'],flush=True)
if __name__=='__main__':main()
