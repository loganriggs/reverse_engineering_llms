"""M64: frozen support/coordinate decomposition with independent NumPy checks."""
from pathlib import Path
import json,hashlib,torch,numpy as np
from torch.nn import functional as F
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 pp=V/'support_change_diagnostic_protocol.json';protocol=json.loads(pp.read_text());rp=V/'state_direction_factorial.json';mp=V/'balanced_response_training_results.json';np_=V/'source_holdout_native_protocol.json'
 for p,k in [(rp,'factorial_sha256'),(mp,'models_sha256'),(np_,'native_protocol_sha256')]:assert sha(p)==protocol[k]
 old=json.loads(rp.read_text());models=json.loads(mp.read_text())['models'];native=json.loads((V/'source_holdout_native.json').read_text());op=V/'support_change_geometry.json';assert not op.exists();dest=A/'emlp_m64_local';dest.mkdir(exist_ok=True);files=[];rows=[];checks=0;maxsum=0.;maxcode=0.;mingap=float('inf')
 states={}
 for e in models:
  p=Path(e['path']);assert sha(p)==e['sha256'];s=torch.load(p,weights_only=True);f=Featurizer(e['kind'],e['seed']).cuda();f.load_state_dict(s['state']);states[e['kind'],e['seed'],e['arm']]=(f,s['mean'].cuda(),s['scale'].cuda())
 for file in old['files']:
  p=Path(file['path']);assert sha(p)==file['sha256'];b=torch.load(p,weights_only=True);entries=[]
  for e in models:
   kind,seed,arm=[e[k] for k in ['kind','seed','arm']];ms=f'{kind}:{seed}:{arm}';f,mu,scale=states[kind,seed,arm];D=F.normalize(f.dictionary,dim=0);enc=[]
   for ep in [0,1]:
    x=(b['writes'][ep].cuda()-mu)/scale;pred,q,code=f(x);a=F.linear(x,f.encoder,f.encoder_bias)
    assert torch.equal(code.cpu(),b['reconstructions'][ms+f'|{ep}']['code'])
    assert torch.equal((pred*scale+mu).cpu(),b['reconstructions'][ms+f'|{ep}']['write'])
    an=x.cpu().double().numpy()@f.encoder.cpu().double().numpy().T+f.encoder_bias.cpu().double().numpy();an_gpu=a.cpu().double().numpy();maxcode=max(maxcode,float(np.max(np.abs(an-an_gpu))));assert np.allclose(an,an_gpu,rtol=2e-5,atol=2e-5)
    scores=np.abs(an) if kind=='scalar' else an**2@f.membership.cpu().double().numpy().T;k=32 if kind=='scalar' else (8 if kind=='block' else 4);order=np.argsort(scores,axis=-1);sorted_scores=np.take_along_axis(scores,order,axis=-1);gap=sorted_scores[...,-k]-sorted_scores[...,-k-1];assert (gap>0).all();mingap=min(mingap,float(gap.min()));selected=np.zeros_like(scores);np.put_along_axis(selected,order[...,-k:],1,axis=-1);qn=selected if kind=='scalar' else ((selected@f.membership.cpu().numpy())>0).astype(float);assert np.array_equal(q.cpu().numpy(),qn);checks+=1;enc.append((a,q))
   for target in [x for x in native['records'] if x['pair_id']==file['pair_id']]:
    ep=target['endpoint'];pos=target['recipient_positions'];dp=target['donor_positions'];ar,qr=enc[ep];ad,qd=enc[1-ep];ar,qr=ar[:,pos],qr[:,pos];ad,qd=ad[:,dp],qd[:,dp];nd=b['writes'][1-ep][:,dp].double()-b['writes'][ep][:,pos].double();full=b['reconstructions'][ms+f'|{1-ep}']['write'][:,dp]-b['reconstructions'][ms+f'|{ep}']['write'][:,pos]
    coeffs={'within':(ad-ar)*qr,'switch':ad*(qd-qr),'donor_within':(ad-ar)*qd,'donor_switch':ar*(qd-qr)};deltas={k:(v@D.T*scale).cpu() for k,v in coeffs.items()};deltas['full']=full
    dn=D.cpu().double().numpy();sc=float(scale);arn=ar.cpu().double().numpy();adn=ad.cpu().double().numpy();qrn=qr.cpu().double().numpy();qdn=qd.cpu().double().numpy();cn={'within':(adn-arn)*qrn,'switch':adn*(qdn-qrn),'donor_within':(adn-arn)*qdn,'donor_switch':arn*(qdn-qrn)}
    for k in cn:assert np.allclose(deltas[k].double().numpy(),cn[k]@dn.T*sc,rtol=2e-4,atol=.003)
    denom=max(float(full.double().square().sum()),1.)
    for pre in ['', 'donor_']:
     err=float((deltas[pre+'within'].double()+deltas[pre+'switch'].double()-full.double()).square().sum())/denom;maxsum=max(maxsum,err);assert err<1e-8
    ident={k:e[k] for k in ['kind','seed','arm']}|dict(key=file['pair_id']+f'|{ep}',family=target['family'],region=target['region']);row=ident|dict(native_energy=float(nd.square().sum()),changed_atoms=int((qr!=qd).sum()),union_atoms=int(((qr+qd)>0).sum()),intersection_atoms=int(((qr*qd)>0).sum()))
    for k,d in deltas.items():row[k+'_energy']=float(d.double().square().sum());row[k+'_native_error']=float((d.double()-nd).square().sum())
    for pre in ['', 'donor_']:row[pre+'cross']=float((deltas[pre+'within'].double()*deltas[pre+'switch'].double()).sum())
    rows.append(row);entries.append(dict(identity=ident,positions=pos,deltas=deltas))
  out=dest/p.name;torch.save(dict(pair_id=file['pair_id'],entries=entries),out);files.append(dict(pair_id=file['pair_id'],path=str(out),sha256=sha(out)));print('LOCAL',len(files),flush=True)
 assert checks==504 and len(rows)==1512
 result=dict(status='passed',protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),factorial_sha256=sha(rp),models_sha256=sha(mp),files=files,rows=rows,gate_checks=checks,min_boundary_gap=mingap,max_encoder_abs_error=maxcode,max_decomposition_relative_squared_error=maxsum,scope='Independent NumPy encoder/support selection and local decoded terms, both anchors; full saved code/write replay. Native responses pending.')
 op.write_text(json.dumps(result,indent=2)+'\n');print('PASS',checks,maxsum,mingap,flush=True)
if __name__=='__main__':main()
