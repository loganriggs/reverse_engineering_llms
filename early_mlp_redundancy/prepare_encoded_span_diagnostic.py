"""M71 teacher-informed nested-span projections; no fitting or native response yet."""
from pathlib import Path
import hashlib,json,torch,numpy as np
from torch.nn import functional as F
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def en(x):return float(np.square(x.astype(np.float64)).sum())
def main():
 torch.set_num_threads(1);pp=V/'encoded_span_diagnostic_protocol.json';protocol=json.loads(pp.read_text());rp=V/'encoded_factorial.json';mp=V/'encoded_change_evaluation_models.json'
 for p,k in [(rp,'factorial_sha256'),(mp,'models_sha256'),(V/'encoded_factorial_audit.json','factorial_audit_sha256')]:assert sha(p)==protocol[k]
 old=json.loads(rp.read_text());models=json.loads(mp.read_text())['models'];native=json.loads((V/'source_holdout_native.json').read_text());sp=Path(native['samples_path']);assert sha(sp)==native['samples_sha256'];ns=torch.load(sp,weights_only=True);op=V/'encoded_span_geometry.json';assert not op.exists();states={};basis_cache={}
 for e in models:
  p=Path(e['path']);assert sha(p)==e['sha256'];s=torch.load(p,weights_only=True);states[e['kind'],e['seed'],e['arm']]=s
 def project(ms,D,indices,target):
  key=(ms,tuple(indices))
  if key not in basis_cache:
   if len(indices):u,sv,_=np.linalg.svd(D[:,indices],full_matrices=False);u=u[:,sv>1e-10*sv[0]] if len(sv) and sv[0]>0 else u[:,:0]
   else:u=np.zeros((D.shape[0],0))
   basis_cache[key]=u
  u=basis_cache[key];pred=u@(u.T@target);orth=np.linalg.norm(u.T@(target-pred))/max(np.linalg.norm(target),1.);idem=np.linalg.norm(u@(u.T@pred)-pred)/max(np.linalg.norm(pred),1.);assert orth<1e-10 and idem<1e-10;return pred,u.shape[1],float(orth),float(idem)
 dest=A/'emlp_m71_local';dest.mkdir(exist_ok=True);rows=[];files=[];gatechecks=0;min_gap=float('inf');maxcode=0.;maxorth=0.;maxidem=0.;maxunionres=0.
 for file in old['files']:
  path=Path(file['path']);assert sha(path)==file['sha256'];blob=torch.load(path,weights_only=True);entries=[]
  for e in models:
   mk=tuple(e[k] for k in ['kind','seed','arm']);ms=':'.join(map(str,mk));s=states[mk];st=s['state'];D=F.normalize(st['dictionary'],dim=0).double().numpy();supports=[]
   for ep in [0,1]:
    # Reproduce float32 normalization, then independently evaluate encoder/gates in double.
    x=((blob['writes'][ep]-s['mean'])/s['scale']).double().numpy();a=x@st['encoder'].double().numpy().T+st['encoder_bias'].double().numpy();scores=np.abs(a) if e['kind']=='scalar' else a*a@st['membership'].double().numpy().T;k=32 if e['kind']=='scalar' else (8 if e['kind']=='block' else 4);order=np.argsort(scores,axis=-1);sorted_scores=np.take_along_axis(scores,order,axis=-1);gap=sorted_scores[...,-k]-sorted_scores[...,-k-1];assert (gap>0).all();min_gap=min(min_gap,float(gap.min()));g=np.zeros_like(scores);np.put_along_axis(g,order[...,-k:],1,axis=-1);q=g>0 if e['kind']=='scalar' else (g@st['membership'].numpy())>0;code=blob['reconstructions'][ms+f'|{ep}']['code'].double().numpy();rel=np.linalg.norm(a*q-code)/max(np.linalg.norm(code),1.);maxcode=max(maxcode,float(rel));assert rel<1e-5 and np.array_equal(q,code!=0);supports.append(q);gatechecks+=1
   for target in [x for x in native['records'] if x['pair_id']==file['pair_id']]:
    ep=target['endpoint'];pos=target['recipient_positions'];dp=target['donor_positions'];nd=(blob['writes'][1-ep].numpy()[:,dp]-blob['writes'][ep].numpy()[:,pos]);assert np.array_equal(nd,ns['deltas'][target['sample_key']]['delta'].numpy());actual=(blob['reconstructions'][ms+f'|{1-ep}']['write'].numpy()[:,dp]-blob['reconstructions'][ms+f'|{ep}']['write'].numpy()[:,pos]);preds={k:np.zeros(nd.shape,dtype=np.float64) for k in ['recipient','union','dictionary']};selected={'recipient':[],'union':[]};ranks={k:[] for k in preds};unionres=0.
    for j,(p,d) in enumerate(zip(pos,dp)):
     ir=np.flatnonzero(supports[ep][0,p]);iu=np.flatnonzero(supports[ep][0,p]|supports[1-ep][0,d]);selected['recipient'].append(ir.tolist());selected['union'].append(iu.tolist())
     for term,inds in [('recipient',ir),('union',iu),('dictionary',np.arange(128))]:
      pred,rank,orth,idem=project(ms,D,inds,nd[0,j].astype(float));preds[term][0,j]=pred;ranks[term].append(rank);maxorth=max(maxorth,orth);maxidem=max(maxidem,idem)
     ua=basis_cache[ms,tuple(iu)];unionres+=en(actual[0,j]-ua@(ua.T@actual[0,j]))
    ne=en(nd);actualerr=en(actual.astype(float)-nd.astype(float));errors={k:en(v-nd.astype(float)) for k,v in preds.items()};tol=1e-6*max(ne,1.);assert errors['dictionary']<=errors['union']+tol and errors['union']<=errors['recipient']+tol and errors['union']<=actualerr+tol;ur=unionres/max(en(actual),1.);maxunionres=max(maxunionres,ur);assert ur<1e-8
    deltas={k:torch.from_numpy(v.astype(np.float32)) for k,v in preds.items()};deltas['actual']=torch.from_numpy(actual.copy());ident={k:e[k] for k in ['kind','seed','arm']}|dict(key=file['pair_id']+f'|{ep}',family=target['family'],region=target['region']);row=ident|dict(native_energy=ne,actual_union_residual_relative=ur,ranks=ranks)
    for k,v in deltas.items():row[k+'_error']=en(v.double().numpy()-nd.astype(float));row[k+'_energy']=en(v.numpy())
    rows.append(row);entries.append(dict(identity=ident,positions=pos,donor_positions=dp,selected=selected,ranks=ranks,deltas=deltas))
  out=dest/path.name;torch.save(dict(pair_id=file['pair_id'],entries=entries),out);files.append(dict(pair_id=file['pair_id'],path=str(out),sha256=sha(out)));print('LOCAL',len(files),flush=True)
 assert len(rows)==1512 and gatechecks==504
 op.write_text(json.dumps(dict(status='passed',source_sha256=sha(Path(__file__)),protocol_sha256=sha(pp),models_sha256=sha(mp),factorial_sha256=sha(rp),rows=rows,files=files,gate_checks=gatechecks,min_gate_gap=min_gap,max_code_relative_error=maxcode,max_projection_orthogonality_error=maxorth,max_idempotence_error=maxidem,max_actual_union_residual_relative=maxunionres,scope='Float64 SVD projection checks; saved applied vectors are float32. All native/actual direction differences match fixed M70 addresses. No native responses yet.'),indent=2)+'\n');print('PASS',gatechecks,maxorth,maxidem,maxunionres,flush=True)
if __name__=='__main__':main()
