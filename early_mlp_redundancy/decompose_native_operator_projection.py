"""M100 exact error decomposition: context projection versus integrable direction projection."""
import json
import torch
from fit_pca_encoder import V,A,sha

def main():
 torch.set_num_threads(1);torch.set_grad_enabled(False);pp=V/'native_operator_projection_decomposition_protocol.json';p=json.loads(pp.read_text())
 for n,h in p['references'].items():assert sha(V/n)==h
 assert p['source_sha256']==sha(__file__)
 ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];r=torch.load(ref['path'],weights_only=True);L,R,D=[r[n].double().cuda() for n in ['L','R','D']];mean=r['mean'].cuda();fit=json.loads((V/'native_operator_metric_fits.json').read_text());assert sha(fit['path'])==fit['sha256'];models=torch.load(fit['path'],weights_only=True)['models'];rows=[];samples=[]
 def j(z,v):return ((z@L.T)*(v@R.T)+(v@L.T)*(z@R.T))@D.T
 def w(z):return ((z@L.T)*(z@R.T))@D.T
 for cohort,stem,ns in [('whole','native_operator_factorial','whole_task_native'),('source','native_operator_source_factorial','source_holdout_native')]:
  result=json.loads((V/(stem+'.json')).read_text());audit=json.loads((V/(stem+'_audit.json')).read_text());assert audit['status']=='passed' and audit['result_sha256']==sha(V/(stem+'.json'));targets=json.loads((V/(ns+'.json')).read_text())['records']
  for file in result['files']:
   assert sha(file['path'])==file['sha256'];blob=torch.load(file['path'],weights_only=True);pid=blob['pair_id']
   for target in targets:
    if target['pair_id']!=pid or target['status']!='evaluated' or target['region']=='identity':continue
    ep=target['endpoint'];pos=target['recipient_positions'];dp=target['donor_positions'];zr=blob['inputs'][ep][0,pos].double().cuda();zd=blob['inputs'][1-ep][0,dp].double().cuda();delta=zd-zr;h=(zr+zd)/2-mean;exact=j(mean+h,delta);finite=w(zd)-w(zr);assert torch.linalg.norm(exact-finite)<=1e-10*torch.linalg.norm(exact)+1e-9;shared=j(mean,delta);target_energy=float(exact.square().sum())
    for basis,m in models.items():
     for rank in [4,16,64]:
      E=m['encoder'][:,:rank].cuda();B=m['decoder'][:,:rank].cuda();ph=(h@E)@B.T;pd=(delta@E)@B.T;anchor=shared+j(ph,delta);consistent=shared+j(ph,pd);a=j(h-ph,delta);b=j(ph,delta-pd);residual=exact-consistent;identity_error=float(torch.linalg.norm(residual-a-b));assert identity_error<=1e-10*float(torch.linalg.norm(exact))+1e-9
      arm=basis+'__'+str(rank);key='native_operator:0:'+arm;qr=blob['reconstructions'][key+f'|{ep}']['write'][0,pos].double().cuda();qd=blob['reconstructions'][key+f'|{1-ep}']['write'][0,dp].double().cuda();applied_error=float(torch.linalg.norm(consistent-(qd-qr)));bound=4*torch.finfo(torch.float32).eps*float(torch.linalg.norm(qr)+torch.linalg.norm(qd))+1e-9;assert applied_error<=bound,(cohort,pid,arm,applied_error,bound)
      rows.append(dict(cohort=cohort,pair_id=pid,endpoint=ep,family=target['family'],region=target['region'],basis=basis,rank=rank,target_energy=target_energy,anchor_error=float((exact-anchor).square().sum()),consistent_error=float(residual.square().sum()),context_term_energy=float(a.square().sum()),direction_term_energy=float(b.square().sum()),twice_cross=float(2*(a*b).sum()),identity_absolute_error=identity_error,applied_difference_error=applied_error,applied_rounding_bound=bound));samples.append(dict(row=len(rows)-1,recipient=zr.cpu(),donor=zd.cpu(),exact=exact.cpu(),anchor=anchor.cpu(),consistent=consistent.cpu(),context_term=a.cpu(),direction_term=b.cpu()))
   print('PAIR',cohort,pid,flush=True)
 assert len(rows)==2448;path=A/'emlp_m100_projection_decomposition.pt';assert not path.exists();torch.save(samples,path);out=dict(status='completed_audit_pending',source_sha256=sha(__file__),protocol_sha256=sha(pp),rows=rows,samples_path=str(path),samples_sha256=sha(path),scope='All9candidates onall272eligible nonidentitytargets acrossbothcohorts. Exactfloat64localfunctionerrors; notnewdownstreamresponses. Context/directionerror terms cancancel; notnativechannelcancellation. Appliedfloat32predictiondifferences withinpredeclaredroundingbound.')
 f=V/'native_operator_projection_decomposition.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print('DONE',len(rows))
if __name__=='__main__':main()
