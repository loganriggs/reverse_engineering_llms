"""H702: data-only sparse rotation followed by coactivation group discovery."""
import torch,json,time,hashlib,itertools,copy
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from feature_structures.models import union_support
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_sparse_rotation_groups'
def main():
 torch.set_num_threads(2);start=time.time();paths={'data':ART/'bilin18_shared_exact_gates_captures.pt','teacher':ART/'feature_structure_synthetic_captures.pt','implementation':ROOT/'scripts/feature_structures/models.py','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);tables=data['tables'];tx=tables['train']['x'];vx=tables['validation']['x'];scale=tx.square().mean();_,_,V=torch.linalg.svd(tx,full_matrices=False);span=V[:8].T;patterns=torch.tensor(list(itertools.product([0.,1.],repeat=4)));triples=list(itertools.combinations(range(8),3));cfg=dict(experiment='H702',hashes=hashes,steps=3000,lr=.01,batch=512,design='Fit8D PCA span on raw uncentered training activations; learn orthogonal rotation minimizing smooth absolute coordinate magnitude(sqrt(z²+1e-6)), choose by validation sparsity. Infer4groups from most frequent top3atom supports among training examples with <=1%energy outside top3. No generating labels in fitting or group selection. Exact16pattern gate inference at0.01primarycost and1e-5sensitivity. Random orthogonal rotation controls for each seed.',limitations='Known8atoms/4groups/3members, orthonormal dictionary assumption, noiseless synthetic teacher, exact inference exponential, repeated exploratory benchmark. Staged unsupervised conditional on these structural priors, not LLM validation.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));states={};records=[];teacher=torch.load(paths['teacher'],weights_only=False);B,M=teacher['true_dictionary'],teacher['true_membership']
 for seed in [61,62,63]:
  torch.manual_seed(seed);base=torch.linalg.qr(torch.randn(8,8)).Q;raw=torch.nn.Parameter(torch.zeros(8,8));opt=torch.optim.Adam([raw],lr=.01);gen=torch.Generator().manual_seed(seed+100);best=float('inf');history=[]
  def dictionary():return span@base@torch.matrix_exp(raw-raw.T)
  for step in range(3001):
   if step%100==0:
    with torch.no_grad():v=float(torch.sqrt((vx@dictionary()).square()+1e-6).mean());history.append(dict(step=step,validation_smooth_l1=v))
    if v<best:best=v;beststep=step;bestD=dictionary().detach().clone()
   if step==3000:break
   ii=torch.randint(len(tx),(512,),generator=gen);loss=torch.sqrt((tx[ii]@dictionary()).square()+1e-6).mean();opt.zero_grad();loss.backward();opt.step()
  for kind,D in [('random_rotation',span@base),('sparse_rotation',bestD)]:
   with torch.no_grad():
    z=tx@D;top=z.square().topk(3,dim=-1).indices;inside=z.square().gather(1,top).sum(-1);eligible=(z.square().sum(-1)-inside)<=.01*z.square().sum(-1);counts={t:0 for t in triples}
    for row in top[eligible].tolist():counts[tuple(sorted(row))]+=1
    chosen=sorted(triples,key=lambda t:(-counts[t],t))[:4];L=torch.zeros(4,8)
    for k,t in enumerate(chosen):L[k,list(t)]=1
    score=(B.T@D).abs();aa,bb=linear_sum_assignment(-score.numpy());perm=max(itertools.permutations(range(4)),key=lambda p:float((L[list(p)][:,bb]==M).float().mean()));agreement=float((L[list(perm)][:,bb]==M).float().mean());evals=[]
    for split in ['validation','heldout']:
     x,g=tables[split]['x'],tables[split]['g'];codes=x@D;q=union_support(patterns,L);pred=torch.einsum('npa,da->npd',codes[:,None,:]*q[None,:,:],D);err=(pred-x[:,None,:]).square().mean(-1)/scale;rows=torch.arange(len(x))
     for cost in [.01,.00001]:
      obj=err+cost*patterns.sum(-1);idx=obj.argmin(-1);gates=patterns[idx];evals.append(dict(split=split,cost=cost,relative_mse=float(err[rows,idx].mean()*scale/x.square().mean()),objective=float(obj[rows,idx].mean()),group_exact_recovery=float((gates[:,list(perm)]==g).all(-1).float().mean()),mean_groups=float(gates.sum(-1).mean()),unique_patterns=int(torch.unique(gates,dim=0).shape[0])))
    rec=dict(seed=seed,kind=kind,selected_step=beststep if kind=='sparse_rotation' else 0,matched_atom_cosine=float(score[aa,bb].mean()),membership_entry_agreement=agreement,group_permutation=perm,eligible_training_examples=int(eligible.sum()),chosen_triples=chosen,chosen_counts=[counts[t] for t in chosen],orthogonality_error=float((D.T@D-torch.eye(8)).abs().max()),validation_smooth_l1=float(torch.sqrt((vx@D).square()+1e-6).mean()),evals=evals,history=history if kind=='sparse_rotation' else []);records.append(rec);states[seed,kind]=dict(dictionary=D,membership=L);print({k:rec[k] for k in ['seed','kind','matched_atom_cosine','membership_entry_agreement','eligible_training_examples','chosen_counts','evals']},flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor(r['matched_atom_cosine'])).item() for r in records),max_orthogonality_error=max(r['orthogonality_error'] for r in records),whole_model_forwards=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and audits['max_orthogonality_error']<1e-5;torch.save(dict(states=states),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
