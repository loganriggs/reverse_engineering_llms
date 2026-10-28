"""M12 fixed block dimensionality and across-seed stability diagnostics."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent;A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');mp=P/'validation/diverse_supervision_results.json';m11=json.loads(mp.read_text());m4=json.loads((P/'validation/natural_token_context.json').read_text());cache=Path(m4['cache']);assert sha(cache)==m4['cache_sha256'];data=torch.load(cache,weights_only=True);Y=data['Y'][1];protocol=dict(source_sha256=sha(Path(__file__)),M11_sha256=sha(mp),models=m11['models'],design='Frozen8models, same64naturalwindows. Train/evaldocs0–31/32–63 reportedseparately. No taskevaluation reads ornativeforwards.32groups4: trained contiguous blockgroups, arbitrary contiguous scalar grouping, plus fixed atompermutation(seed1200+modelseed) groupingcontrol. Actualcodesunchanged. Conditionalactive covariance measured in decoder-outputmetric via QR, participationrank andr90. Crossseed Hungarianmatch by decoder-subspace overlap only; train/eval supportJaccard reportedafter. Global128dimsubspace overlap separately. One signed ambientpermutation null(seed12012) plus regrouping control; descriptive references not significance tests.',limits='32training/evaldocuments are reuseddevelopment. Dimensionality, stablegeometry and coactivation do not prove semanticmanifold orcausalrole. Groupoccupancy differsforblock/scalar/shuffled; report ratherthan equate. No hyperparameter selection.')
 pp=P/'validation/block_structure_protocol.json';pp.write_text(json.dumps(protocol,indent=2)+'\n');cached={};rows=[];outstates=[]
 for entry in m11['models']:
  path=Path(entry['path']);assert sha(path)==entry['sha256'];s=torch.load(path,weights_only=True);m=Featurizer(entry['kind'],entry['seed']).cuda();m.load_state_dict(s['state'])
  with torch.no_grad():_,support,code=m((Y.cuda()-s['mean'].cuda())/s['scale'].cuda());D=torch.nn.functional.normalize(m.dictionary,dim=0).cpu().double().numpy();code=code.cpu().double().numpy();support=support.cpu().numpy()
  key=f"{entry['kind']}:{entry['seed']}:{entry['arm']}";groups={'contiguous':np.arange(128).reshape(32,4),'shuffled':np.random.default_rng(1200+entry['seed']).permutation(128).reshape(32,4)};cached[key]=dict(D=D,code=code,support=support,groups=groups);outstates.append(dict(key=key,D=torch.tensor(D),code=torch.tensor(code),support=torch.tensor(support),groups={k:torch.tensor(v) for k,v in groups.items()}))
  for grouping,indices in groups.items():
   for split,idx in [('train',np.arange(4096)),('evaluation',np.arange(4096,8192))]:
    for g,atoms in enumerate(indices):
     a=code[idx][:,atoms];active=np.any(support[idx][:,atoms]>0,axis=1);n=int(active.sum());Q,R=np.linalg.qr(D[:,atoms],mode='reduced');selected=a[active];centered=selected-selected.mean(0) if n else selected;cov=centered.T@centered/max(n,1);outputcov=R@cov@R.T;eig=np.linalg.eigvalsh(outputcov).clip(0);energy=float(eig.sum());rank=float(energy**2/np.sum(eig**2)) if energy>1e-20 else 0.;r90=int(np.searchsorted(np.cumsum(eig[::-1])/energy,.9))+1 if energy>1e-20 else 0
     rows.append(dict(key=key,kind=entry['kind'],seed=entry['seed'],arm=entry['arm'],grouping=grouping,split=split,group=g,active_rows=n,occupancy=n/len(idx),conditional_all4_fraction=float(np.mean((support[idx][:,atoms]>0).sum(1)[active]==4)) if n else 0.,decoder_min_singular=float(np.linalg.svd(D[:,atoms],compute_uv=False)[-1]),conditional_participation_rank=rank,conditional_r90=r90,output_covariance=outputcov.tolist(),code_covariance=cov.tolist(),output_eigenvalues=eig.tolist()))
 summaries=[]
 for entry in m11['models']:
  for grouping in ['contiguous','shuffled']:
   for split in ['train','evaluation']:
    rr=[x for x in rows if all(x[k]==entry[k] for k in ['kind','seed','arm']) and x['grouping']==grouping and x['split']==split];summaries.append({k:entry[k] for k in ['kind','seed','arm']}|dict(grouping=grouping,split=split,occupied_groups=sum(x['active_rows']>0 for x in rr),median_active_rows=float(np.median([x['active_rows'] for x in rr])),mean_occupancy=float(np.mean([x['occupancy'] for x in rr])),median_participation_rank=float(np.median([x['conditional_participation_rank'] for x in rr])),median_r90=float(np.median([x['conditional_r90'] for x in rr])),mean_conditional_all4_fraction=float(np.mean([x['conditional_all4_fraction'] for x in rr]))))
 stability=[];rng=np.random.default_rng(12012);perm=rng.permutation(1152);sign=rng.choice([-1.,1.],size=1152)
 for kind in ['scalar','block']:
  for arm in ['natural_only_KL','mixed_natural_task_KL']:
   ka=f'{kind}:71:{arm}';kb=f'{kind}:72:{arm}';a,b=cached[ka],cached[kb];ga=np.linalg.qr(a['D'],mode='reduced')[0];gb=np.linalg.qr(b['D'],mode='reduced')[0];global_overlap=float(np.square(ga.T@gb).sum()/128)
   for grouping in ['contiguous','shuffled','ambient_null']:
    name='contiguous' if grouping=='ambient_null' else grouping;ia=a['groups'][name];ib=b['groups'][name];Db=b['D'][perm]*sign[:,None] if grouping=='ambient_null' else b['D'];qa=np.stack([np.linalg.qr(a['D'][:,ii],mode='reduced')[0] for ii in ia]);qb=np.stack([np.linalg.qr(Db[:,ii],mode='reduced')[0] for ii in ib]);overlap=np.square(np.einsum('gdi,hdj->ghij',qa,qb)).sum((2,3))/4;ai,bi=linear_sum_assignment(-overlap);supports={}
    for split,ix in [('train',np.arange(4096)),('evaluation',np.arange(4096,8192))]:
     sa=np.stack([np.any(a['support'][ix][:,ii]>0,axis=1) for ii in ia],1);sb=np.stack([np.any(b['support'][ix][:,ii]>0,axis=1) for ii in ib],1);intersection=np.logical_and(sa[:,ai],sb[:,bi]).sum(0);union=np.logical_or(sa[:,ai],sb[:,bi]).sum(0);jaccard=np.divide(intersection,union,out=np.zeros(32,dtype=float),where=union>0);supports[split]=dict(mean_jaccard=float(jaccard.mean()),per_group_jaccard=jaccard.tolist(),union_counts=union.tolist())
    stability.append(dict(kind=kind,arm=arm,grouping=grouping,global_subspace_overlap=global_overlap,mean_matched_overlap=float(overlap[ai,bi].mean()),overlap_matrix=overlap.tolist(),assignment=bi.tolist(),supports=supports))
 dest=A/'emlp_m12_structure_inputs.pt';torch.save(outstates,dest);out=dict(status='completed',protocol_sha256=sha(pp),rows=rows,summaries=summaries,stability=stability,inputs_path=str(dest),inputs_sha256=sha(dest),model_forwards=0,featurizer_examples=8*8192);(P/'validation/block_structure.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([x for x in summaries if x['split']=='evaluation' and x['grouping']=='contiguous'],indent=2));print('STABILITY',[(x['kind'],x['arm'],x['grouping'],round(x['global_subspace_overlap'],3),round(x['mean_matched_overlap'],3),round(x['supports']['evaluation']['mean_jaccard'],3)) for x in stability],flush=True)
if __name__=='__main__':main()
