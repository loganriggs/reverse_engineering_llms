"""M103 fixed PCA64 span, four coordinate systems, training inputs only."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,sha
from coordinate_gauge_diagnostic import rotation,components,summaries


def main():
    torch.set_num_threads(1)
    protocol_path=V/'native_coordinate_gauge_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    assert protocol['source_sha256']==sha(__file__)
    assert protocol['helper_sha256']==sha(V.parent/'coordinate_gauge_diagnostic.py')
    for name,h in protocol['receipts'].items():assert sha(V/name)==h
    rf=json.loads((V/'native_input_reference.json').read_text());pf=json.loads((V/'native_operator_metric_fits.json').read_text())
    assert sha(rf['path'])==rf['sha256'] and sha(pf['path'])==pf['sha256']
    audit=json.loads((V/'native_operator_metric_audit.json').read_text());assert audit['status']=='passed' and audit['fit_sha256']==sha(V/'native_operator_metric_fits.json')
    s=torch.load(rf['path'],weights_only=True);m=torch.load(pf['path'],weights_only=True)['models']['euclidean']
    B=m['encoder'][:,:64].double();assert torch.equal(B,m['decoder'][:,:64].double())
    assert torch.allclose(B.T@B,torch.eye(64,dtype=torch.float64),atol=1e-10,rtol=1e-10)
    indices=s['indices'].numpy();selected=indices[np.linspace(0,len(indices)-1,128,dtype=np.int64)];assert len(set(map(int,selected)))==128
    pairs=selected.reshape(64,2);zall=s['z'].double();mean=s['mean'].double().cuda();L=s['L'].double().cuda();R=s['R'].double().cuda();D=s['D'].double().cuda();base=B.cuda()
    F=lambda x:((x@L.T)*(x@R.T))@D.T
    J=lambda z,v:((z@L.T)*(v@R.T)+(v@L.T)*(z@R.T))@D.T
    maxima={};rows=[];numpy_checks=0
    def check(name,a,b):
        scale=1+float(torch.linalg.vector_norm(b));error=float(torch.linalg.vector_norm(a-b))/scale
        maxima[name]=max(maxima.get(name,0.),error);assert error<1e-9,(name,error)
    rotations=[('PCA',np.eye(64))]+[(f'rotation_{seed}',rotation(64,seed)) for seed in protocol['seeds']]
    for case,(i,j) in enumerate(pairs):
        z=zall[int(i)].cuda();delta=(zall[int(j)]-zall[int(i)]).cuda();fixed=(delta@base)@base.T
        response=F(z+fixed)-F(z)
        mixed=lambda x,b:F(x)-F((x-mean)-((x-mean)@b)@b.T)
        reference=mixed(z,base);reference_donor=mixed(z+delta,base)
        for name,Q in rotations:
            basis=base@torch.from_numpy(Q).cuda();parts=(delta@basis)[:,None]*basis.T;d=parts.sum(0)
            midpoint=J(z+d/2,parts);single=J(z,parts)+F(parts);interaction=F(d)-F(parts).sum(0)
            check('projected_change',d,fixed);check('midpoint_sum',midpoint.sum(0),response)
            check('individual_finite',single,F(z+parts)-F(z));check('interaction_remainder',single.sum(0)+interaction,response)
            check('candidate_recipient',mixed(z,basis),reference);check('candidate_donor',mixed(z+delta,basis),reference_donor)
            arrays=[x.cpu().numpy() for x in [parts,midpoint,single,response,interaction]]
            stats=summaries(*arrays)
            if case in protocol['numpy_oracle_cases']:
                oracle=components(s['L'].double().numpy(),s['R'].double().numpy(),s['D'].double().numpy(),z.cpu().numpy(),delta.cpu().numpy(),basis.cpu().numpy())
                for actual,expected in zip(arrays,oracle):
                    err=float(np.linalg.norm(actual-expected)/(1+np.linalg.norm(expected)));assert err<1e-9;maxima['numpy_oracle']=max(maxima.get('numpy_oracle',0.),err)
                numpy_checks+=1
            rows.append(dict(case=case,recipient_index=int(i),donor_index=int(j),basis=name,**stats))
        if case%16==15:print('CHECKED',case+1,flush=True)
    aggregates=[]
    for name,_ in rotations:
        rr=[x for x in rows if x['basis']==name]
        entry=dict(basis=name,cases=len(rr))
        for field in ['input_participation','midpoint_participation','single_participation']:
            values=[x[field] for x in rr if x[field] is not None];entry[field+'_mean']=float(np.mean(values));entry[field+'_min']=min(values);entry[field+'_max']=max(values)
        en=sum(x['response_energy'] for x in rr)
        entry['interaction_pooled_relative_energy']=sum(x['interaction_energy'] for x in rr)/en
        entry['midpoint_pooled_net_over_sum']=en/sum(x['midpoint_energy_sum'] for x in rr)
        aggregates.append(entry)
    out=dict(status='passed',source_sha256=sha(__file__),protocol_sha256=sha(protocol_path),rows=rows,aggregates=aggregates,max_scaled_errors=maxima,numpy_oracle_conditions=numpy_checks,scope=protocol['scope'])
    dest=V/'native_coordinate_gauge.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(aggregates,indent=2),flush=True)
if __name__=='__main__':main()
