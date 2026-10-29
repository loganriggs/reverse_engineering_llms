import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from quartic_cross_gram import cross_gram
from larger_hidden_transfer import dictionary,values,sphere_factor,ridge,target_moments
src=Path('results/scaling_grid');out=Path('results/larger_hidden_transfer');out.mkdir(exist_ok=False);records=[]
for gi,d in [(0,16),(2,32)]:
 for kind in ['shared','independent']:
  for rep in range(3):
   key=f'g{gi}_{kind}_r{rep}';folder=src/key;teacher=torch.load(folder/'teacher.pt',weights_only=True);data=np.load(folder/'training.npz');x=data['adaptation_inputs'];y=data['adaptation_targets']
   for model in ['shared','separate','shared_budget','oracle_source']:
    start=time.perf_counter();file=folder/('teacher.pt' if model=='oracle_source' else model+'.pt');state=torch.load(file,weights_only=True)
    F,T=dictionary(state,range(12) if model=='oracle_source' else None)
    G=cross_gram(F,F)*sphere_factor(d);C,E=target_moments(F,teacher,d);X=values(F,x)
    # Polynomial teacher-target replay on stored sphere inputs.
    target=np.column_stack([values(dictionary(teacher,[t])[0],x)@dictionary(teacher,[t])[1][:,0] for t in range(12,16)])
    np.testing.assert_allclose(target,y,rtol=1e-8,atol=1e-10)
    saved=dict(factors=F,source_map=T,target_cross=C,target_energy=E,gram_diagonal=np.diag(G))
    for representation in ['source','hidden']:
     if representation=='source':GG=T.T@G@T;CC=T.T@C;XX=X@T
     else:GG,CC,XX=G,C,X
     norm=np.sqrt(np.maximum(np.diag(GG),1e-300));Z=XX/norm
     for n in [32,128]:
      b=ridge(Z[:n],y[:n]);coef=b/norm[:,None];error=np.sum(coef*(GG@coef),axis=0)-2*np.sum(coef*CC,axis=0)+E
      assert min(error/E)>-1e-7
      normal=np.linalg.norm(Z[:n].T@(Z[:n]@b-y[:n])+n*.001*b)/max(1.,np.linalg.norm(Z[:n].T@y[:n]));assert normal<1e-7
      saved[f'{representation}_{n}']=coef
      records.append(dict(key=key,d=d,teacher=kind,model=model,representation=representation,n=n,features=len(norm),relative_mse=(error/E).tolist(),normal_relative=float(normal),checkpoint_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),data_sha256=hashlib.sha256((folder/'training.npz').read_bytes()).hexdigest()))
    if model!='oracle_source':
     old=np.load(folder/f'{model}_evaluation.npz');GG=T.T@G@T;CC=T.T@C
     historical=[]
     for b in old['B']:
      coef=b/old['feature_norm'][:,None];err=np.sum(coef*(GG@coef),axis=0)-2*np.sum(coef*CC,axis=0)+E;historical.append(err/E)
     saved['historical_relative_mse']=np.array(historical)
    np.savez_compressed(out/f'{key}_{model}.npz',**saved)
    (out/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(key,model,'dimension',len(F),'seconds',round(time.perf_counter()-start,2),flush=True)
