"""Coordinate-invariant local and pooled operator spectra on the M103 span."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,sha
from coordinate_gauge_diagnostic import rotation


def spectrum(g):
    e=np.linalg.eigvalsh(g)[::-1];assert e[-1]>=-1e-10*e[0];e=np.maximum(e,0);total=e.sum()
    return dict(effective_rank=float(total**2/(e@e)),remaining_energy={str(k):float(e[k:].sum()/total) for k in [1,4,16,64]},eigenvalues=e.tolist())


def main():
    torch.set_num_threads(1)
    protocol=json.loads((V/'native_gauge_spectrum_protocol.json').read_text());assert protocol['source_sha256']==sha(__file__)
    for name,h in protocol['inputs'].items():assert sha(V/name)==h
    r=json.loads((V/'native_input_reference.json').read_text());p=json.loads((V/'native_operator_metric_fits.json').read_text())
    assert sha(r['path'])==r['sha256'] and sha(p['path'])==p['sha256']
    s=torch.load(r['path'],weights_only=True);B=torch.load(p['path'],weights_only=True)['models']['euclidean']['encoder'][:,:64].double().cuda()
    L=s['L'].double().cuda();R=s['R'].double().cuda();D=s['D'].double().cuda();Z=s['z'].double().cuda()
    result=json.loads((V/'native_coordinate_gauge.json').read_text());cases=[x for x in result['rows'] if x['basis']=='PCA'];assert len(cases)==64
    rotations=[torch.from_numpy(rotation(64,seed)).cuda() for seed in [10301,10302,10303]]
    total=np.zeros((64,64));rows=[];maximum=0.
    for row in cases:
        z=Z[row['recipient_index']];delta=Z[row['donor_index']]-z;h=z+((delta@B)@B.T)/2
        K=((h@L.T)*(B.T@R.T)+(h@R.T)*(B.T@L.T))@D.T
        gram=(K@K.T).cpu().numpy();total+=gram;stat=spectrum(gram)
        singular=torch.linalg.svdvals(K).square().cpu().numpy()
        err=float(np.linalg.norm(singular-np.array(stat['eigenvalues']))/np.linalg.norm(singular));maximum=max(maximum,err);assert err<1e-10
        for Q in rotations:
            direct=(((h@L.T)*((B@Q).T@R.T)+(h@R.T)*((B@Q).T@L.T))@D.T)
            expected=Q.T@K
            assert torch.allclose(direct,expected,atol=1e-9,rtol=1e-9)
            ev=torch.linalg.eigvalsh(direct@direct.T).flip(0).cpu().numpy()
            err=float(np.linalg.norm(ev-singular)/np.linalg.norm(singular));maximum=max(maximum,err);assert err<1e-10
        rows.append(dict(case=row['case'],**stat))
    pooled=spectrum(total/len(rows))
    out=dict(status='passed',source_sha256=sha(__file__),protocol_sha256=sha(V/'native_gauge_spectrum_protocol.json'),rows=rows,pooled=pooled,mean_local_effective_rank=float(np.mean([x['effective_rank'] for x in rows])),min_local_effective_rank=min(x['effective_rank'] for x in rows),max_local_effective_rank=max(x['effective_rank'] for x in rows),max_relative_spectrum_check_error=maximum,scope=protocol['scope'])
    f=V/'native_gauge_spectrum.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ['rows','pooled']});print('POOLED',{k:v for k,v in pooled.items() if k!='eigenvalues'})
if __name__=='__main__':main()
