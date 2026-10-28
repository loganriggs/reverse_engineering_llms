"""H584: discovery-fit agreement write subspaces against live full-swap references."""
import json,hashlib,time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_agreement_write_subspace'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_locked_writer_lexical_transfer.json';prior=json.loads(src.read_text());saved=torch.load(ART/'bilin18_locked_writer_lexical_transfer_cache.pt',weights_only=False);ex={k:e for k,e in saved['examples'].items() if e['family']=='agreement'};cap=saved['captures'];keys=[k for k,e in ex.items() if e['variant']<2];ranks=[1,2,4,8]
 cfg=dict(experiment='H584',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),layer=5,site=-1,ranks=ranks,random_seeds=list(range(5)),fit='Uncentered SVD of eight discovery plural-minus-singular Attention5 final-position write differences. Nuisance basis from sixteen same-answer discovery differences. Mean direction and a fixed mean write also tested. Fit and save before running any intervention.',design='All discovery and lexical-transfer agreement endpoints; counterfactual and nuisance donors. Full swap and zero controls. PCA projections; nuisance and five random subspaces norm-matched to PCA projected perturbation at the same rank per case. Fixed mean tested only for counterfactual swaps.',fidelity='Internal relative error; KL(full-swap || approximation); margin and intended-group NLL errors against full swap; native-to-intervened KL and donor movement separately. Low-dimensional representation is not an upstream generator: projected coefficients use the actual donor difference.',scope='Eight nouns per split, one wording, whole attention output, paired directions dependent. No universal feature or canonical-basis claim.',criteria='Descriptive compact fidelity: mean absolute margin error <=10% of mean absolute full-swap effect AND mean reference KL <=10% of full-vs-native KL; do not convert passing into a semantic label.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));disc=sorted(k[0] for k,e in ex.items() if e['split']=='discovery' and k[1]==0);mat=torch.stack([cap[i,1]['attn',5][-1]-cap[i,0]['attn',5][-1] for i in disc]);nm=torch.stack([cap[i,v+2]['attn',5][-1]-cap[i,v]['attn',5][-1] for i in disc for v in (0,1)]);_,sv,V=torch.linalg.svd(mat,full_matrices=False);_,ns,N=torch.linalg.svd(nm,full_matrices=False);mean=mat.mean(0);mdir=mean/mean.norm();random={seed:torch.linalg.qr(torch.randn(mat.shape[1],8,generator=torch.Generator().manual_seed(seed)))[0].T for seed in cfg['random_seeds']};bases=dict(pca=V,nuisance=N,mean=mean,random=random,singular_values=sv,nuisance_singular_values=ns);bp=ART/f'{STEM}_locked_bases.pt';torch.save(bases,bp);bh=hashlib.sha256(bp.read_bytes()).hexdigest();print('LOCKED BASES',bh,flush=True)
 model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 jobs=defaultdict(list);deltas={};result=dict(config=cfg,basis_sha256=bh,discovery_energy=[float(sv[:r].square().sum()/sv.square().sum()) for r in ranks],interventions=[],summary={},status='running')
 for k in keys:
  e=ex[k]
  for dtype in ('counterfactual','nuisance'):
   donor=(k[0],1-k[1] if dtype=='counterfactual' else k[1]+2);d=cap[donor]['attn',5][-1]-cap[k]['attn',5][-1];vecs={'full':d,'zero':torch.zeros_like(d),'mean_projection':mdir*(mdir@d)}
   if dtype=='counterfactual':vecs['fixed_mean']=mean*(1 if k[1]==0 else -1)
   for r in ranks:
    pd=V[:r].T@(V[:r]@d);vecs[f'pca_{r}']=pd
    nd=N[:r].T@(N[:r]@d);vecs[f'nuisance_matched_{r}']=nd*pd.norm()/nd.norm().clamp_min(1e-20)
    for seed,R in random.items():
     rd=R[:r].T@(R[:r]@d);vecs[f'random{seed}_matched_{r}']=rd*pd.norm()/rd.norm().clamp_min(1e-20)
   for mode,v in vecs.items():jobs[len(e['ids'])].append((k,dtype,mode,v));deltas[k,dtype,mode]=v
 print('LOCKED INTERVENTIONS',sum(map(len,jobs.values())),flush=True);outputs={}
 for T,jj in jobs.items():
  for st in range(0,len(jj),24):
   bb=jj[st:st+24];zz,_=run(model,torch.tensor([ex[k]['ids'] for k,dt,m,d in bb],device='cuda'),patch=(5,torch.stack([d for k,dt,m,d in bb])))
   for ix,(k,dt,mode,d) in enumerate(bb):outputs[k,dt,mode]=zz[ix].cpu()
 def kl(a,b):return float((a.exp()*(a-b)).sum())
 for T,jj in jobs.items():
  for k,dt,mode,d in jj:
   z=outputs[k,dt,mode];ref=outputs[k,dt,'full'];native=saved['logits'][k];lp=z.log_softmax(0);rp=ref.log_softmax(0);np_=native.log_softmax(0);gs=ex[k]['groups'];ans=gs[k[1]];margin=measure(z,gs)['margin'];rm=measure(ref,gs)['margin'];nm_=measure(native,gs)['margin'];fd=deltas[k,dt,'full'];result['interventions'].append(dict(id=k[0],variant=k[1],split=ex[k]['split'],donor_type=dt,mode=mode,internal_relative_error=float((d-fd).norm()/fd.norm()),retained_norm_ratio=float(d.norm()/fd.norm()),reference_kl=kl(rp,lp),native_kl=kl(np_,lp),full_native_kl=kl(rp,np_),margin_error=margin-rm,full_margin_delta=rm-nm_,margin_delta=margin-nm_,donor_directed_delta=(margin-nm_)*(1 if k[1]==1 else -1),group_nll_error=float(torch.logsumexp(rp[ans],0)-torch.logsumexp(lp[ans],0))))
 for split in ('discovery','transfer'):
  for dt in ('counterfactual','nuisance'):
   cc=[c for c in result['interventions'] if c['split']==split and c['donor_type']==dt];den=np.mean([abs(c['full_margin_delta']) for c in cc if c['mode']=='full']);kd=np.mean([c['full_native_kl'] for c in cc if c['mode']=='full'])
   for mode in sorted({c['mode'] for c in cc}):
    vv=[c for c in cc if c['mode']==mode];result['summary'][f'{split}_{dt}_{mode}']=dict(n=len(vv),internal_relative_error=float(np.mean([c['internal_relative_error'] for c in vv])),reference_kl=float(np.mean([c['reference_kl'] for c in vv])),relative_reference_kl=float(np.mean([c['reference_kl'] for c in vv])/kd),relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/den),mean_donor_directed_delta=float(np.mean([c['donor_directed_delta'] for c in vv])),group_nll_abs_error=float(np.mean([abs(c['group_nll_error']) for c in vv])))
 old={(c['id'],c['variant'],c['mode']):c for c in prior['interventions'] if c['family']=='agreement' and c['operation']=='attn_5' and c['site']==-1};fulls=[c for c in result['interventions'] if c['mode']=='full'];zeros=[c for c in result['interventions'] if c['mode']=='zero'];result['audits']=dict(full_replay_max_margin_error=max(abs(c['margin_delta']-old[c['id'],c['variant'],c['donor_type']]['full']['margin_delta']) for c in fulls),zero_replay_max_margin_error=max(abs(c['margin_delta']) for c in zeros),pca_orthogonality_error=float((V@V.T-torch.eye(len(V))).abs().max()),matched_norm_max_error=max(float(abs(v.norm()-deltas[k,dt,f'pca_{mode.rsplit("_",1)[1]}'].norm())) for (k,dt,mode),v in deltas.items() if 'matched' in mode),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ('internal_relative_error','reference_kl','margin_delta')),basis_unchanged=bh==hashlib.sha256(bp.read_bytes()).hexdigest(),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(result['audits'],flush=True)
if __name__=='__main__':main()
