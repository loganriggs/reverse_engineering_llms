"""H703: fixed staged pipeline on fresh structures, noise and nonorthogonal atoms."""
import torch,json,time,hashlib,itertools
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from feature_structures.models import union_support
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_sparse_rotation_robustness'
def main():
 torch.set_num_threads(2);start=time.time();paths={'previous_runner':ROOT/'scripts/run_bilin18_sparse_rotation_groups.py','implementation':ROOT/'scripts/feature_structures/models.py','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};patterns=torch.tensor(list(itertools.product([0.,1.],repeat=4)));triples=list(itertools.combinations(range(8),3));trainsets=[s for n in [1,2] for s in itertools.combinations(range(4),n)];heldsets=[s for n in [3,4] for s in itertools.combinations(range(4),n)];cfg=dict(experiment='H703',hashes=hashes,steps=3000,lr=.01,batch=512,design='Three fresh teachers with random4distinct3atom groups covering8atoms, paired clean/noisy/nonorthogonal conditions. Same H702 orthogonal sparse-rotation and top3support frequency extraction,1%tail-energy threshold, exactgatecost0.01primary/1e-5sensitivity. Train/val singletonpairs,holdtriples/allout. Noise isotropic SD5%trainingcleanRMS; nonorthogonality D=normalize(Q@(I+0.4cyclicshift)). One same initialization per teacher across conditions. No label fitting or selection.',limitations='Known sizes, orthogonal fitted dictionary, three teacher seeds not large statistical benchmark. One initialization perteacher/condition. Component intervention score measures recovered synthetic factor contributions, not actual LLM causality.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];saved={}
 for teacher_seed in [7031,7032,7033]:
  gen=torch.Generator().manual_seed(teacher_seed);Q=torch.linalg.qr(torch.randn(16,8,generator=gen)).Q
  while True:
   ids=torch.randperm(len(triples),generator=gen)[:4];M=torch.zeros(4,8)
   for g,k in enumerate(ids):M[g,list(triples[int(k)])]=1
   if (M.sum(0)>0).all():break
  def sample(sets,n):
   gates=torch.zeros(n,4);which=torch.randint(len(sets),(n,),generator=gen)
   for i,k in enumerate(which):gates[i,list(sets[int(k)])]=1
   code=torch.randn(n,8,generator=gen)*(gates@M>0);return code,gates
  raw={split:sample(sets,n) for split,sets,n in [('train',trainsets,4096),('validation',trainsets,2048),('heldout',heldsets,2048)]};noise={sp:torch.randn(len(v[0]),16,generator=gen) for sp,v in raw.items()}
  for condition in ['clean','noise_5pct','nonorthogonal']:
   B=Q if condition!='nonorthogonal' else torch.nn.functional.normalize(Q@(torch.eye(8)+.4*torch.roll(torch.eye(8),1,0)),dim=0);clean={sp:v[0]@B.T for sp,v in raw.items()};sigma=.05*clean['train'].square().mean().sqrt() if condition=='noise_5pct' else 0.;x={sp:clean[sp]+sigma*noise[sp] for sp in raw};tx,vx=x['train'],x['validation'];scale=tx.square().mean();_,_,V=torch.linalg.svd(tx,full_matrices=False);span=V[:8].T;torch.manual_seed(teacher_seed+100);base=torch.linalg.qr(torch.randn(8,8)).Q;param=torch.nn.Parameter(torch.zeros(8,8));opt=torch.optim.Adam([param],lr=.01);bgen=torch.Generator().manual_seed(teacher_seed+200);best=float('inf')
   def dictionary():return span@base@torch.matrix_exp(param-param.T)
   for step in range(3001):
    if step%100==0:
     with torch.no_grad():val=float(torch.sqrt((vx@dictionary()).square()+1e-6).mean())
     if val<best:best=val;beststep=step;D=dictionary().detach().clone()
    if step==3000:break
    ii=torch.randint(len(tx),(512,),generator=bgen);loss=torch.sqrt((tx[ii]@dictionary()).square()+1e-6).mean();opt.zero_grad();loss.backward();opt.step()
   with torch.no_grad():
    z=tx@D;top=z.square().topk(3,dim=-1).indices;total=z.square().sum(-1);inside=z.square().gather(1,top).sum(-1);eligible=(total-inside)<=.01*total;counts={t:0 for t in triples}
    for row in top[eligible].tolist():counts[tuple(sorted(row))]+=1
    chosen=sorted(triples,key=lambda t:(-counts[t],t))[:4];L=torch.zeros(4,8)
    for k,t in enumerate(chosen):L[k,list(t)]=1
    score=(B.T@D).abs();aa,bb=linear_sum_assignment(-score.numpy());perm=max(itertools.permutations(range(4)),key=lambda p:float((L[list(p)][:,bb]==M).float().mean()));hx=x['heldout'];codes=hx@D;truthcode,gates=raw['heldout'];q=union_support(patterns,L);pred=torch.einsum('npa,da->npd',codes[:,None,:]*q[None,:,:],D);err=(pred-hx[:,None,:]).square().mean(-1)/scale;rows=torch.arange(len(hx));evals=[]
    classes={}
    for atom in range(8):classes.setdefault(tuple(M[:,atom].tolist()),[]).append(atom)
    blocks=list(classes.values());subspaces=[]
    for group in range(4):
     trueQ=torch.linalg.qr(B[:,M[group]>0]).Q;learnQ=torch.linalg.qr(D[:,L[perm[group]]>0]).Q;subspaces.append(float((trueQ.T@learnQ).square().sum()/3))
    for cost in [.01,.00001]:
     idx=(err+cost*patterns.sum(-1)).argmin(-1);s=patterns[idx];recon=pred[rows,idx];used=codes*q[idx];effects=[]
     for a,b in zip(aa,bb):
      effect=used[:,b,None]*D[:,b];true=truthcode[:,a,None]*B[:,a];effects.append(float((effect-true).square().sum()/true.square().sum()))
     blockerrors=[]
     for atoms in blocks:
      actual=used[:,bb[atoms]]@D[:,bb[atoms]].T;truth=truthcode[:,atoms]@B[:,atoms].T;blockerrors.append(float((actual-truth).square().sum()/truth.square().sum()))
     evals.append(dict(support_exact_recovery=float((q[idx][:,bb]==(gates@M>0)).all(-1).float().mean()),mean_coactivation_block_effect_error=sum(blockerrors)/len(blockerrors),max_coactivation_block_effect_error=max(blockerrors),cost=cost,observed_relative_mse=float((recon-hx).square().sum()/hx.square().sum()),clean_relative_mse=float((recon-clean['heldout']).square().sum()/clean['heldout'].square().sum()),group_exact_recovery=float((s[:,list(perm)]==gates).all(-1).float().mean()),mean_component_effect_error=sum(effects)/len(effects),max_component_effect_error=max(effects)))
    rec=dict(coactivation_blocks=blocks,distinct_teacher_unions=int(torch.unique((patterns@M>0).int(),dim=0).shape[0]),group_subspace_overlap=subspaces,teacher_seed=teacher_seed,condition=condition,selected_step=beststep,matched_atom_cosine=float(score[aa,bb].mean()),membership_entry_agreement=float((L[list(perm)][:,bb]==M).float().mean()),eligible_training_examples=int(eligible.sum()),chosen_counts=[counts[t] for t in chosen],teacher_membership=M.tolist(),teacher_max_offdiag_cosine=float(((B.T@B)-torch.eye(8)).abs().max()),orthogonality_error=float((D.T@D-torch.eye(8)).abs().max()),evals=evals);records.append(rec);saved[teacher_seed,condition]=dict(dictionary=D,membership=L,teacher=B,true_membership=M);print(rec,flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor(e['clean_relative_mse'])).item() for r in records for e in r['evals']),max_fitted_orthogonality_error=max(r['orthogonality_error'] for r in records),whole_model_forwards=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and audits['max_fitted_orthogonality_error']<1e-5;torch.save(dict(states=saved),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
