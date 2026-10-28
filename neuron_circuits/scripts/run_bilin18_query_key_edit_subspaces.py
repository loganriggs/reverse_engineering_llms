"""H742: paired q2/k2 edit subspaces with name-held-out and cross-format tests."""
import json,time
from pathlib import Path
import torch,numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_coactivation_group_swaps import sha
from run_bilin18_recall_query_key_cross import coefficient
STEM='bilin18_query_key_edit_subspaces'
def rotary_values(position,dim=128):
 inv=1.0/(10000**(torch.arange(0,dim,2).float()/dim));freq=torch.tensor(float(position),dtype=torch.float32)*inv;return freq.cos().bfloat16().double(),freq.sin().bfloat16().double()
def rotate(x,position,inverse=False):
 c,s=rotary_values(position,x.shape[-1]);a,b=x.double().chunk(2,-1)
 if inverse:return torch.cat([(c*a-s*b)/(c*c+s*s),(s*a+c*b)/(c*c+s*s)],-1)
 return torch.cat([c*a+s*b,-s*a+c*b],-1)
def projected_edit(receiver,donor,basis,renorm=False):
 result=receiver+(donor-receiver)@basis@basis.T
 if renorm:result=result*(receiver.norm()/result.norm().clamp_min(1e-12))
 return result
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'qk':ART/'bilin18_recall_query_key_cross_captures.pt','previous':ART/'bilin18_recall_query_key_cross.json','positions':ART/'bilin18_recall_operation_cross_captures.pt','position_results':ART/'bilin18_recall_operation_cross.json','source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py'),'runner':Path(__file__),'coefficient':ROOT/'scripts/run_bilin18_recall_query_key_cross.py'};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['qk'],weights_only=False);qk=data['qk'];pairs=data['matched_context_pairs'];examples=torch.load(paths['positions'],weights_only=False)['examples'];prior=json.loads(paths['previous'].read_text());geom=json.loads(paths['position_results'].read_text())['geometry'];oldalpha={(r['id'],r['receiver']):r['alpha'] for r in geom};lam=4.1875
 cfg=dict(experiment='H742',hashes=hashes,fit_names=[0,1,2,3],evaluation_names=[4,5,6,7],design='ReusedH740/H741pairedcontexts. Remove actualbf16-cached RoPE usingexact2x2 inverse, fituncenteredSVD ofbinding->copy q2/k2 edits onfirst4namepairs only. Separatequery/key bases. Scopespooledbothformats(ranks1/2/4/8/16),narrativeonly andlistonly(ranks1/2/4/8). Evaluateallreciprocaledits, separateheldoutnames andcross-formattransfer. Threeindependentrandomorthogonalsubspace controls perrank. Rawprojectededits andreceiver-norm-preserving versions. RestoreRoPEatrecipientpositions, holdnativefactor1fixed. Targetisexactfactor2swap, notallfourcomponentswap.',limitations='Supervisedpairededitdiagnostic, notunsupervisedmanifolddiscovery. Namesheldoutfrombasisfit only; panelpreviouslyinspected. Separateq/k rankr meansr directionsperstate, notone rDjointfeature. Crossformat removesrotarycoordinateframe difference but retainsdifferentrelativepositions/nativecontexts. No downstreammodelrunhere; coefficientfidelity alone insufficientforfinalclaim.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));unrot={};roundtrip=[]
 for key,parts in qk.items():
  e=examples[key];positions={'q2':len(e['ids'])-1,'k2':e['source_position']};unrot[key]={}
  for name,pos in positions.items():
   x=parts[name].double();y=rotate(x,pos,True);unrot[key][name]=y;roundtrip.append(float((rotate(y,pos)-x).norm()/x.norm()))
 bases={};training={}
 for scope in ['pooled','narrative','list']:
  edits={'q2':[],'k2':[]}
  for key,donor in pairs:
   e=examples[key];i=int(key[0].rsplit('-',1)[1])
   if e['family']!='entity_binding' or i>=4 or (scope!='pooled' and e['format']!=scope):continue
   for name in edits:edits[name].append(unrot[donor][name]-unrot[key][name])
  for name,rows in edits.items():
   X=torch.stack(rows);u,s,vh=torch.linalg.svd(X,full_matrices=False);training[scope,name]=dict(samples=len(X),singular_values=s.tolist(),numerical_rank=int((s>s[0]*1e-10).sum()))
   for rank in ([1,2,4,8,16] if scope=='pooled' else [1,2,4,8]):bases[scope,rank,None,name]=vh[:rank].T
 for seed in [7421,7422,7423]:
  gen=torch.Generator().manual_seed(seed)
  for name in ['q2','k2']:
   Q=torch.linalg.qr(torch.randn(128,128,generator=gen,dtype=torch.double)).Q
   for rank in [1,2,4,8,16]:bases['random',rank,seed,name]=Q[:,:rank]
 for name in ['q2','k2']:bases['identity',128,None,name]=torch.eye(128,dtype=torch.double)
 variants=sorted({(scope,rank,seed) for scope,rank,seed,name in bases},key=str);records=[];identityerrors=[];orth=[]
 for key,donor in pairs:
  e=examples[key];i=int(key[0].rsplit('-',1)[1]);source=unrot[key];target=unrot[donor];nativeparts={n:v.double() for n,v in qk[key].items()};fullparts={**nativeparts,'q2':qk[donor]['q2'].double(),'k2':qk[donor]['k2'].double()};alpha=float(coefficient(nativeparts,lam));full=float(coefficient(fullparts,lam));positions={'q2':len(e['ids'])-1,'k2':e['source_position']}
  for scope,rank,seed in variants:
   for renorm in [False,True]:
    projected={};energies={};norms={}
    for name in ['q2','k2']:
     B=bases[scope,rank,seed,name];orth.append(float((B.T@B-torch.eye(rank,dtype=torch.double)).abs().max()));v=projected_edit(source[name],target[name],B,renorm);projected[name]=rotate(v,positions[name]);delta=target[name]-source[name];energies[name]=float(((delta@B)**2).sum()/delta.square().sum().clamp_min(1e-20));norms[name]=float(v.norm()/source[name].norm())
    predicted=float(coefficient({**nativeparts,**projected},lam));rec=dict(id=key[0],receiver=key[1],family=e['family'],format=e['format'],name_index=i,split='fit_names' if i<4 else 'heldout_names',scope=scope,rank=rank,seed=seed,renorm=renorm,native_alpha=alpha,target_alpha=full,predicted_alpha=predicted,query_edit_energy_fraction=energies['q2'],key_edit_energy_fraction=energies['k2'],query_norm_ratio=norms['q2'],key_norm_ratio=norms['k2']);records.append(rec)
    if scope=='identity' and not renorm:identityerrors.append(abs(predicted-full))
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},rotary_inverse_roundtrip_max_relative=max(roundtrip),basis_orthogonality_max_abs=max(orth),full_rank_coefficient_replay_max_abs=max(identityerrors),finite=all(np.isfinite(r['predicted_alpha']) for r in records),whole_model_forwards=0,test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(roundtrip)<1e-10 and max(orth)<1e-10 and max(identityerrors)<1e-10;torch.save(dict(bases=bases,unrotated=unrot,training=training),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,training=[dict(scope=k[0],component=k[1],**v) for k,v in training.items()],records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
