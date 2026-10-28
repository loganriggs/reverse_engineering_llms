"""H589: cross propagated final residual responses with recipient final-readout contexts."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_number_terminal_response_cross'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_number_fixed_dose.json';old=json.loads(src.read_text());bp=ART/'bilin18_agreement_write_subspace_locked_bases.pt';b=torch.load(bp,weights_only=False);axis=b['pca'][0]*torch.sign(b['pca'][0]@b['mean']);scale=float(axis@b['mean']);ex={}
 for stem in ('bilin18_number_subject_attractor','bilin18_number_fronted_attractor'):ex.update(torch.load(ART/f'{stem}_cache.pt',weights_only=False)['examples'])
 frames=['preposition','relative_clause','fronted_preposition'];doses=[.0625,.5];cfg=dict(experiment='H589',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),basis_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),doses=doses,discovery_amplitude=scale,design='For each matched lexical/number quadruple, capture final residual response to the same signed Attention5 perturbation in each frame. Cross each donor final-response vector with each recipient native final residual (3x3), then apply the unchanged model final RMS, unembedding, tanh and answer-group readout. Diagonals must reproduce H588. Offdiagonals are terminal residual interventions, not natural prompts.',scope='Separates frame-dependent propagation into final residual from baseline-dependent terminal readout, conditional on this response interchange. Terminal readout includes final RMS, tanh and group aggregation. Does not uniquely assign mediation to earlier blocks.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 buckets=defaultdict(list)
 for k,e in ex.items():buckets[len(e['ids'])].append(k)
 residual={};native_logits={}
 for T,kk in buckets.items():
  for st in range(0,len(kk),24):
   bb=kk[st:st+24];ids=torch.tensor([ex[k]['ids'] for k in bb],device='cuda')
   for dose in [0.]+[sgn*d for d in doses for sgn in (-1,1)]:
    zz,cache=run(model,ids,capture=True,patch=None if dose==0 else (5,(axis*scale*dose).expand(len(bb),-1)))
    for ix,k in enumerate(bb):residual[k,dose]=cache['last'][ix].clone()
    if dose==0:
     for ix,k in enumerate(bb):native_logits[k]=zz[ix].cpu()
 print('CAPTURED',len(residual),flush=True);idx={(e['kind'],e['item'],e['subject_number'],e['attractor_number'],e['frame']):k for k,e in ex.items()};jobs=[]
 for k,e in ex.items():
  for donorframe in frames:
   dk=idx[e['kind'],e['item'],e['subject_number'],e['attractor_number'],donorframe]
   for dose in doses:
    for sign in (-1,1):jobs.append((k,dk,sign*dose))
 result=dict(config=cfg,crossings=[],responses=[],summary={},decomposition=[],status='running')
 for st in range(0,len(jobs),48):
  bb=jobs[st:st+48];x=torch.stack([residual[k,0.]+residual[dk,dose]-residual[dk,0.] for k,dk,dose in bb]).cuda();zz=30*torch.tanh(model.lm_head(F.rms_norm(x,(x.shape[-1],))).float()/30)
  for ix,(k,dk,dose) in enumerate(bb):
   e=ex[k];nz=native_logits[k].to(zz);lp=zz[ix].log_softmax(0);np_=nz.log_softmax(0);ans=e['groups'][e['subject_number']];result['crossings'].append(dict(id=k,donor_id=dk,kind=e['kind'],recipient_frame=e['frame'],donor_frame=ex[dk]['frame'],dose=dose,plural_shift=measure(nz,e['groups'])['margin']-measure(zz[ix],e['groups'])['margin'],kl=float((np_.exp()*(np_-lp)).sum()),group_nll_delta=float(torch.logsumexp(np_[ans],0)-torch.logsumexp(lp[ans],0))))
 lookup={(c['id'],c['donor_frame'],c['dose']):c for c in result['crossings']}
 for k,e in ex.items():
  for donorframe in frames:
   for dose in doses:
    p=lookup[k,donorframe,dose];n=lookup[k,donorframe,-dose];result['responses'].append(dict(id=k,kind=e['kind'],recipient_frame=e['frame'],donor_frame=donorframe,dose=dose,central_plural_response=(p['plural_shift']-n['plural_shift'])/(2*dose),mean_kl=(p['kl']+n['kl'])/2))
 for kind in ['regular','irregular']:
  for dose in doses:
   for rf in frames:
    for df in frames:
     cc=[c for c in result['responses'] if (c['kind'],c['dose'],c['recipient_frame'],c['donor_frame'])==(kind,dose,rf,df)];result['summary'][f'{kind}_{dose}_{rf}_from_{df}']=dict(n=len(cc),mean_central_plural_response=float(np.mean([c['central_plural_response'] for c in cc])),positive_fraction=float(np.mean([c['central_plural_response']>0 for c in cc])),mean_kl=float(np.mean([c['mean_kl'] for c in cc])))
   for target in frames[1:]:
    def v(r,d):return result['summary'][f'{kind}_{dose}_{r}_from_{d}']['mean_central_plural_response']
    pp=v('preposition','preposition');pf=v('preposition',target);fp=v(target,'preposition');ff=v(target,target);result['decomposition'].append(dict(kind=kind,dose=dose,target=target,total_change=ff-pp,propagation_symmetric_contribution=((pf-pp)+(ff-fp))/2,terminal_context_symmetric_contribution=((fp-pp)+(ff-pf))/2,interaction=ff-fp-pf+pp))
 oldlookup={(c['id'],c['dose']):c for c in old['interventions'] if c['direction']=='number'};replay=[abs(c['plural_shift']-oldlookup[c['id'],c['dose']]['plural_shift']) for c in result['crossings'] if c['recipient_frame']==c['donor_frame']];result['audits']=dict(diagonal_cases=len(replay),diagonal_H588_margin_max_error=max(replay),symmetric_decomposition_max_error=max(abs(c['total_change']-c['propagation_symmetric_contribution']-c['terminal_context_symmetric_contribution']) for c in result['decomposition']),finite=all(np.isfinite(c['plural_shift']) and np.isfinite(c['kl']) for c in result['crossings']),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),basis_unchanged=cfg['basis_sha256']==hashlib.sha256(bp.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));torch.save(dict(residual=residual,examples=ex),ART/f'{STEM}_residuals.pt');print(result['audits'],flush=True)
if __name__=='__main__':main()
