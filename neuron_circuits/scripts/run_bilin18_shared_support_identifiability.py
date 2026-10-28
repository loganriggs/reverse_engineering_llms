"""H688: analytic support-coverage audit of trained synthetic shared dictionaries."""
import json,time,hashlib,itertools
from pathlib import Path
import torch
from torch.nn import functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_shared_support_identifiability'
@torch.no_grad()
def main():
 start=time.time();paths={'capture':ART/'feature_structure_synthetic_captures.pt','results':ART/'feature_structure_synthetic.json','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};cap=torch.load(paths['capture'],weights_only=False);src=json.loads(paths['results'].read_text());B=cap['true_dictionary'].double();M=cap['true_membership'].double();train=src['config']['train_supports'];test=src['config']['test_supports']
 def support(groups):return (M[groups].sum(0)>0).double()
 co=torch.zeros(8,8,dtype=torch.bool)
 for group in train:q=support(group).bool();co|=q[:,None]&q[None,:]
 unseen=~co;cfg=dict(experiment='H688',hashes=hashes,design='Analytic expected Gaussian reconstruction error of H687 known-membership models, by training/heldout support. Inspect encoder coupling between atom pairs never jointly active. Teacher-basis diagnostic zeroes those entries without fitting. Compare closed-form errors and fresh MonteCarlo.',limitations='Uses synthetic teacher dictionary/supports. Diagnostic is not an available unsupervised fix or learned recovery. Fixed known-membership models only.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];mcmax=0.
 for seed in [11,12,13]:
  st=cap['states']['shared_known_membership',seed];D=F.normalize(st['dictionary'].double(),dim=0);E=st['coordinate_encoder.weight'].double();eb=st['coordinate_encoder.bias'].double();bias=st['bias'].double();coupling=E@B;fixed=E-(coupling*unseen)@B.T;aligned=(D.T@B).abs();entry=dict(seed=seed,min_diagonal_atom_alignment=float(aligned.diag().min()),mean_diagonal_atom_alignment=float(aligned.diag().mean()),unseen_coupling_norm=float((coupling*unseen).norm()),seen_offdiagonal_coupling_norm=float((coupling*co*(1-torch.eye(8))).norm()),cells=[])
  for split,groupsets in [('train',train),('heldout',test)]:
   for group in groupsets:
    q=support(group);truth=B*q;cell=dict(split=split,groups=group)
    for name,enc in [('original',E),('zero_unseen',fixed)]:
     A=(D*q)@enc;offset=(D*q)@eb+bias;error=((A-torch.eye(16))@truth).square().sum()+offset.square().sum();cell[name]=float(error/q.sum())
    entry['cells'].append(cell)
  gen=torch.Generator().manual_seed(688+seed);x=torch.randn(20000,8,generator=gen,dtype=torch.float64)@B.T;pred=(x@E.T+eb)@D.T+bias;mc=float((pred-x).square().mean()/x.square().mean());expected=next(c['original'] for c in entry['cells'] if len(c['groups'])==4);entry['all_groups_monte_carlo']=mc;entry['all_groups_analytic']=expected;mcmax=max(mcmax,abs(mc-expected));records.append(entry)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor(c['original'])).item() for r in records for c in r['cells']),monte_carlo_max_absolute_difference=mcmax,whole_model_forwards=0);assert mcmax<.003
 (ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,unobserved_atom_pairs=unseen.nonzero().tolist(),records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(json.dumps(dict(pairs=unseen.nonzero().tolist(),records=[{k:v for k,v in r.items() if k!='cells'} for r in records],audits=audits)))
if __name__=='__main__':main()
