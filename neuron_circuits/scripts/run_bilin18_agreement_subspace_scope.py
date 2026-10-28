"""H585: frozen agreement basis across wording and task scopes."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_agreement_subspace_scope'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_agreement_write_subspace.json';bp=ART/'bilin18_agreement_write_subspace_locked_bases.pt';bases=torch.load(bp,weights_only=False);V=bases['pca'];R=bases['random'];N=bases['nuisance'];s582=torch.load(ART/'bilin18_task_boundary_factorial_cache.pt',weights_only=False);s583=torch.load(ART/'bilin18_locked_writer_lexical_transfer_cache.pt',weights_only=False);examples={k:e for k,e in s582['examples'].items() if e['family']=='agreement' and k[1]<2};examples.update({k:e for k,e in s583['examples'].items() if e['family']!='agreement' and k[1]<2});cfg=dict(experiment='H585',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),basis_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),design='Freeze H584 agreement basis. Test all four H582 agreement wording cells and H583 discovery-selected cells for other tasks. Opposite-answer swaps at Attention5 final position only. All token-valid rows retained; no basis refit or scale tuning.',ranks=[1,4,8],controls='Zero, full, nuisance rank1 and five random rank1 directions norm-matched to PCA rank1.',scope='Previously evaluated prompts, new projection interventions. Other-task gates inherited, including failures. Wording variations change final token (window vs usually). Cross-task projection success would concern only these donor differences, not a full model canonical form.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 buckets=defaultdict(list);writes={};native={}
 for k,e in examples.items():buckets[len(e['ids'])].append(k)
 for T,kk in buckets.items():
  for st in range(0,len(kk),24):
   bb=kk[st:st+24];cap={}
   def hook(m,args,out):cap['write']=out[0][:,-1].cpu()
   handle=model.blocks[5].attn.register_forward_hook(hook)
   try:z,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:handle.remove()
   for ix,k in enumerate(bb):writes[k]=cap['write'][ix].clone();native[k]=z[ix].cpu()
 jobs=defaultdict(list);vectors={}
 for k,e in examples.items():
  d=writes[k[0],1-k[1]]-writes[k];vv={'full':d,'zero':torch.zeros_like(d)}
  for rank in (1,4,8):vv[f'pca_{rank}']=V[:rank].T@(V[:rank]@d)
  target=vv['pca_1'].norm();nd=N[:1].T@(N[:1]@d);vv['nuisance_matched_1']=nd*target/nd.norm().clamp_min(1e-20)
  for seed,B in R.items():
   rd=B[:1].T@(B[:1]@d);vv[f'random{seed}_matched_1']=rd*target/rd.norm().clamp_min(1e-20)
  for mode,v in vv.items():jobs[len(e['ids'])].append((k,mode,v));vectors[k,mode]=v
 print('LOCKED',sum(map(len,jobs.values())),flush=True);out={}
 for T,jj in jobs.items():
  for st in range(0,len(jj),24):
   bb=jj[st:st+24];z,_=run(model,torch.tensor([examples[k]['ids'] for k,m,v in bb],device='cuda'),patch=(5,torch.stack([v for k,m,v in bb])))
   for ix,(k,m,v) in enumerate(bb):out[k,m]=z[ix].cpu()
 result=dict(config=cfg,interventions=[],summary={},competence={},status='running')
 for k,e in examples.items():
  nm=measure(native[k],e['groups'])['margin'];ref=out[k,'full'];rp=ref.log_softmax(0);np_=native[k].log_softmax(0);rm=measure(ref,e['groups'])['margin'];fullkl=float((rp.exp()*(rp-np_)).sum())
  for mode in [m for kk,m in out if kk==k]:
   z=out[k,mode];lp=z.log_softmax(0);margin=measure(z,e['groups'])['margin'];d=vectors[k,'full'];result['interventions'].append(dict(id=k[0],variant=k[1],family=e['family'],condition=e['condition'],split=e['split'],mode=mode,internal_relative_error=float((vectors[k,mode]-d).norm()/d.norm()),reference_kl=float((rp.exp()*(rp-lp)).sum()),full_native_kl=fullkl,margin_error=margin-rm,full_margin_delta=rm-nm,margin_delta=margin-nm,donor_directed_delta=(margin-nm)*(1 if k[1]==1 else -1)))
 cells=sorted({(e['family'],e['condition'],e['split']) for e in examples.values()})
 for fam,cond,split in cells:
  cc=[c for c in result['interventions'] if (c['family'],c['condition'],c['split'])==(fam,cond,split)];ee=[(k,e) for k,e in examples.items() if (e['family'],e['condition'],e['split'])==(fam,cond,split)];ms={k:measure(native[k],e['groups']) for k,e in ee};pairs=[k[0] for k,e in ee if k[1]==0];both=float(np.mean([ms[i,0]['margin']>0 and ms[i,1]['margin']<0 for i in pairs]));top=float(np.mean([ms[k]['top1'][k[1]] for k,e in ee]));key=f'{fam}_{cond}_{split}';result['competence'][key]=dict(pairs=len(pairs),both_endpoint_choice=both,full_vocab_top1=top,passes=bool(len(pairs)>=4 and both>=.75 and top>=.5));den=np.mean([abs(c['full_margin_delta']) for c in cc if c['mode']=='full']);kd=np.mean([c['full_native_kl'] for c in cc if c['mode']=='full'])
  for mode in sorted({c['mode'] for c in cc}):
   vv=[c for c in cc if c['mode']==mode];result['summary'][key+'_'+mode]=dict(n=len(vv),internal_relative_error=float(np.mean([c['internal_relative_error'] for c in vv])),reference_kl=float(np.mean([c['reference_kl'] for c in vv])),relative_reference_kl=float(np.mean([c['reference_kl'] for c in vv])/kd),relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/den),mean_donor_directed_delta=float(np.mean([c['donor_directed_delta'] for c in vv])))
 old=json.loads(src.read_text());lookup={(c['id'],c['variant'],c['mode']):c for c in old['interventions'] if c['donor_type']=='counterfactual'};repeat=[abs(c['margin_delta']-lookup[c['id'],c['variant'],c['mode']]['margin_delta']) for c in result['interventions'] if (c['id'],c['variant'],c['mode']) in lookup];nativeerr=[]
 for k,e in examples.items():
  oldz=s582['logits'][k] if e['family']=='agreement' else s583['logits'][k];nativeerr.append(float((native[k]-oldz).abs().max()))
 result['audits']=dict(native_logit_replay_max_error=max(nativeerr),H584_replay_max_margin_error=max(repeat),repeated_interventions=len(repeat),zero_margin_max_error=max(abs(c['margin_delta']) for c in result['interventions'] if c['mode']=='zero'),finite=all(np.isfinite(c['reference_kl']) and np.isfinite(c['internal_relative_error']) for c in result['interventions']),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),basis_unchanged=cfg['basis_sha256']==hashlib.sha256(bp.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(result['audits'],flush=True)
if __name__=='__main__':main()
