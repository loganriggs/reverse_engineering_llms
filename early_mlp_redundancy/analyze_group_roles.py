"""M13 train-only functional matches, evaluated on source-separated families."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cosine_matrix(a,b):return (a@b.T)/np.maximum(np.linalg.norm(a,axis=1)[:,None]*np.linalg.norm(b,axis=1)[None,:],1e-20)
def main():
 ip=P/'group_roles.json';r=json.loads(ip.read_text());protocol=json.loads((P/'group_roles_protocol.json').read_text());assert sha(P/'group_roles_protocol.json')==r['protocol_sha256'];geo=json.loads((P/'block_structure.json').read_text());families={part:sorted({x['family'] for x in r['records'] if x['split']==part}) for part in ['train','evaluation']};matrices={};descriptive=[]
 for kind,grouping in [('scalar','contiguous'),('block','contiguous'),('block','shuffled')]:
  for seed in [71,72]:
   for part in ['train','evaluation']:
    rows=[x for x in r['records'] if x['kind']==kind and x['seed']==seed and x['grouping']==grouping and x['split']==part];M=np.zeros((32,len(families[part])));E=M.copy();cost=M.copy();KL=M.copy()
    for g in range(32):
     for j,f in enumerate(families[part]):
      rr=[x for x in rows if x['group']==g and x['family']==f];assert rr;M[g,j]=np.mean([x['margin_loss'] for x in rr]);E[g,j]=np.mean([x['removed_relative_energy'] for x in rr]);cost[g,j]=np.mean([x['mean_removed_active_coordinates'] for x in rr]);KL[g,j]=np.mean([x['mean_KL'] for x in rr])
    key=f'{kind}:{grouping}:{seed}:{part}';matrices[key]=dict(margin=M.tolist(),energy=E.tolist(),cost=cost.tolist(),KL=KL.tolist());select=np.divide(np.abs(M).max(1),np.abs(M).sum(1),out=np.zeros(32),where=np.abs(M).sum(1)>0);cor=float(np.corrcoef(np.abs(M).ravel(),E.ravel())[0,1]);descriptive.append(dict(kind=kind,grouping=grouping,seed=seed,split=part,mean_KL=float(KL.mean()),mean_energy=float(E.mean()),mean_active_cost=float(cost.mean()),median_max_family_abs_share=float(np.median(select)),abs_margin_energy_correlation=cor))
 matches=[]
 for kind,grouping in [('scalar','contiguous'),('block','contiguous'),('block','shuffled')]:
  a=np.array(matrices[f'{kind}:{grouping}:71:train']['margin']);b=np.array(matrices[f'{kind}:{grouping}:72:train']['margin']);at=np.array(matrices[f'{kind}:{grouping}:71:evaluation']['margin']);bt=np.array(matrices[f'{kind}:{grouping}:72:evaluation']['margin']);C=cosine_matrix(a,b);_,functional=linear_sum_assignment(-C);geometric=np.array(next(x['assignment'] for x in geo['stability'] if x['kind']==kind and x['arm']=='mixed_natural_task_KL' and x['grouping']==grouping));random=np.random.default_rng(13013).permutation(32)
  for method,assignment in [('functional',functional),('geometric',geometric),('random',random)]:
   trainb=b[assignment];testb=bt[assignment];alpha=np.maximum(np.divide(np.sum(a*trainb,1),np.sum(a*a,1),out=np.zeros(32),where=np.sum(a*a,1)>0),0);prediction=alpha[:,None]*at;den=np.sum(testb*testb);error=float(np.sum((prediction-testb)**2)/den);flatcos=float(np.sum(prediction*testb)/max(np.linalg.norm(prediction)*np.linalg.norm(testb),1e-20));percos=np.diag(cosine_matrix(at,testb));matches.append(dict(kind=kind,grouping=grouping,method=method,assignment=assignment.tolist(),train_amplitudes=alpha.tolist(),train_mean_matched_cosine=float(C[np.arange(32),assignment].mean()),evaluation_relative_squared_error=error,evaluation_flattened_cosine=flatcos,evaluation_mean_group_cosine=float(percos.mean()),evaluation_group_cosines=percos.tolist(),evaluation_target_energy=float(den)))
 out=dict(status='completed',source_sha256=sha(Path(__file__)),group_roles_sha256=sha(ip),families=families,signatures=matrices,descriptive=descriptive,matches=matches,protocol='All32groups included. Assignment andnonnegativeamplitudes fitted to10familytrainingmargin signatures only; evaluation6families notusedforfit. Relativeerror compares prediction ofseed72effects tozero-effectprediction denominator. Arelativeerror>1meansworsethanzero. No groupselection or significance claim.')
 (P/'group_role_matching.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([{k:x[k] for k in ['kind','grouping','method','train_mean_matched_cosine','evaluation_relative_squared_error','evaluation_flattened_cosine','evaluation_mean_group_cosine']} for x in matches],indent=2))
if __name__=='__main__':main()
