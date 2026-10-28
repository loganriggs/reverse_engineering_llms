"""H949 scale-invariant, split-sensitive contribution complexity pilot."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha
from run_bilin18_real_causal_distillation import Student
from run_bilin18_anchored_product_structure import greedy
from pathlib import Path
import torch,numpy as np,json,time
import torch.nn.functional as F
S='bilin18_product_complexity_training';LAMBDAS=[0.,.01,.1,1.]
def contribution_energy(m,anchors):
 L=F.normalize(m.left,dim=0);R=F.normalize(m.right,dim=0);outenergy=m.out.square().sum(1);energy=torch.zeros(64,device=L.device)
 for p in anchors.values():
  u=F.rms_norm(p['x'],(1152,));v=F.rms_norm(p['x']+p['delta'],(1152,));dc=(v@L)*(v@R)-(u@L)*(u@R);energy+=.5*dc.square().sum(0)*outenergy/p['effect'].square().sum()
 return energy
@torch.no_grad()
def concentration(m,anchors,ev):
 L=F.normalize(m.left,dim=0);R=F.normalize(m.right,dim=0);O=m.out;rows=[];orders={};u=F.rms_norm(ev['inputs'],(1152,));v=F.rms_norm(ev['inputs']+ev['deltas'],(1152,));ec=(v@L)*(v@R)-(u@L)*(u@R);full=ec@O
 for fam,p in anchors.items():
  u=F.rms_norm(p['x'],(1152,));v=F.rms_norm(p['x']+p['delta'],(1152,));dc=(v@L)*(v@R)-(u@L)*(u@R);G=((dc.T@dc)*(O@O.T)).double();order=greedy(G);orders[fam]=order
  for k in [1,2,4,8,16,32,64]:
   ix=order[:k];part=ec[:,ix]@O[ix];de=float((dc[:,ix]@O[ix]-dc@O).norm()/(dc@O).norm().clamp_min(1e-12))
   for kind in ['full','rank1']:
    ii=[n for n,j in enumerate(ev['jobs']) if j['family']==fam and j['arm']==kind];truth=ev['true_effects'][ii];rows.append(dict(family=fam,intervention=kind,k=k,discovery_student_error=de,heldout_student_error=float((part[ii]-full[ii]).norm()/full[ii].norm().clamp_min(1e-12)),heldout_teacher_error=float((part[ii]-truth).norm()/truth.norm())))
 return rows,orders

def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'student':ROOT/'scripts/run_bilin18_real_causal_distillation.py','greedy':ROOT/'scripts/run_bilin18_anchored_product_structure.py','natural':ART/'bilin18_real_causal_distillation_targets.pt','normalizers':ART/'bilin18_real_causal_distillation_fits.pt','initial':ART/'bilin18_supervised_relay_anchors_fits.pt','anchors':ART/'bilin18_supervised_relay_anchors_anchors.pt','evaluation':ART/'bilin18_distilled_relay_mechanisms_captures.pt'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H949','hashes':hashes,'lambdas':LAMBDAS,'seeds':[0,1],'steps':2000,'objective':'H946 natural native/random effects + equal-task discovery-anchor effectMSE; add lambda*sum_j[(E_j+1e-12)^.25-1e-3], E_j=mean across tasks of sum squared whole-product finite contributions / sum squared teacher effects. Dimensionless, invariant to factor/output scale gauges.','why_quarter_power':'A sum of contribution norms is unchanged by splitting a product into identical smaller copies. The quarter power of squared norm is concave and penalizes such splitting; smoothing avoids singular derivatives. It is a sparsity surrogate, not exact product count.','scope':'Complexity charged only on known discovery-anchor interventions, not all states or model storage. Same64products/parameters for all fits. Train from same-seed H946 anchor_effects checkpoint, fresh Adam .003 and matched batches; allfinalfits retained. No evaluation-based selection.','evaluation':'Natural independent probes and H849 taskeffects; discovery-greedy wholeproduct subsets1..64; no coefficient refit. Full-output causal fidelity remains required before promotion.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');nat=torch.load(paths['natural'],weights_only=False);norm=torch.load(paths['normalizers'],weights_only=False);init=torch.load(paths['initial'],weights_only=False);anc=torch.load(paths['anchors'],weights_only=False)['anchors'];ev=torch.load(paths['evaluation'],weights_only=False);xx=nat['train_x'].cuda();dd=nat['train_delta'].cuda();yy=nat['train_y'].cuda();ee=nat['train_effect'].cuda();ag={f:{k:v.cuda() for k,v in p.items()} for f,p in anc.items()};nd=norm['native_den'];ed=norm['effect_den'];mu=norm['mean'];records=[];curves=[];models={};preds={};subsets=[];orders={};penalties=[]
 # Algebraic scale/split controls independent of model optimization.
 e=torch.tensor([.01,.1,1.],dtype=torch.double);p=lambda z:(z+1e-12).pow(.25)-1e-3;split_ratio=2*p(e/4)/p(e);assert bool((split_ratio>1.4).all())
 for seed in [0,1]:
  for lam in LAMBDAS:
   m=Student(seed,mu).cuda();m.load_state_dict(init['models']['anchor_effects',seed]);opt=torch.optim.Adam(m.parameters(),lr=.003);gen=torch.Generator().manual_seed(94900+seed)
   for step in range(2000):
    ii=torch.randint(len(xx),(128,),generator=gen).cuda();a=m(xx[ii]);b=m(xx[ii]+dd[ii]);base=(a-yy[ii]).square().mean()/nd+(b-a-ee[ii]).square().mean()/ed;anchorloss=torch.zeros((),device='cuda')
    for q in ag.values():anchorloss+=.5*(m(q['x']+q['delta'])-m(q['x'])-q['effect']).square().mean()/q['effect_den']
    E=contribution_energy(m,ag);complexity=((E+1e-12).pow(.25)-1e-3).sum();loss=base+anchorloss+lam*complexity;assert bool(torch.isfinite(loss));opt.zero_grad();loss.backward();opt.step()
    if step%250==0 or step==1999:curves.append(dict(seed=seed,lam=lam,step=step+1,base_loss=float(base.detach()),anchor_loss=float(anchorloss.detach()),complexity=float(complexity.detach())))
   m=m.cpu();models[lam,seed]=m.state_dict()
   with torch.no_grad():
    E=contribution_energy(m,anc);penalties.append(dict(lam=lam,seed=seed,complexity=float(((E+1e-12).pow(.25)-1e-3).sum()),contribution_magnitudes=E.sqrt().tolist()))
    for name,q in nat['panels'].items():
     y=m(q['x']);e=m(q['x']+q['delta'])-y;records.append(dict(lam=lam,seed=seed,domain='natural',panel=name,native_error_ratio=float((y-q['y']).norm()/(q['y']-mu).norm()),effect_error_ratio=float((e-q['effect']).norm()/q['effect'].norm())))
    pe=m(ev['inputs']+ev['deltas'])-m(ev['inputs']);preds[lam,seed]=pe
    for fam in ['agreement','sentiment']:
     for kind in ['full','rank1']:
      ii=[n for n,j in enumerate(ev['jobs']) if j['family']==fam and j['arm']==kind];records.append(dict(lam=lam,seed=seed,domain=fam,panel=kind,effect_error_ratio=float((pe[ii]-ev['true_effects'][ii]).norm()/ev['true_effects'][ii].norm())))
    rr,order=concentration(m,anc,ev);subsets.extend([r|dict(lam=lam,seed=seed) for r in rr]);orders[lam,seed]=order
   print('DONE',lam,seed,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'split_duplicate_penalty_ratios':split_ratio.tolist(),'models':len(models),'finite':all(np.isfinite(r['effect_error_ratio']) for r in records),'whole_model_forwards':0,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'];torch.save(dict(models=models,predicted_effects=preds,orders=orders),ART/f'{S}_fits.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,curves=curves,concentration=subsets,penalties=penalties,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(records,audits,flush=True)
if __name__=='__main__':main()
