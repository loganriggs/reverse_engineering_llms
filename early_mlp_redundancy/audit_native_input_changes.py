"""M77 independent NumPy polarization and explicit shared Jacobian checks."""
import json
import numpy as np
import torch
from fit_pca_encoder import sha,V
P=V.parent

def main():
    torch.set_num_threads(1);rp=V/'native_input_changes.json';result=json.loads(rp.read_text());assert result['source_sha256']==sha(P/'prepare_native_input_changes.py')
    ref=json.loads((V/'native_input_reference.json').read_text());assert result['reference_sha256']==sha(V/'native_input_reference.json') and sha(ref['path'])==ref['sha256'];state=torch.load(ref['path'],weights_only=True)
    cache=json.loads((V/'natural_token_context.json').read_text());assert sha(cache['cache'])==cache['cache_sha256'];data=torch.load(cache['cache'],weights_only=True);indices=np.array([int(i) for i in data['unique_indices'] if int(data['documents'][i])<32]);assert np.array_equal(indices,state['indices'].numpy());mean=state['z'].numpy().astype(float)[indices].mean(0);assert np.allclose(mean,state['mean'].numpy(),rtol=1e-13,atol=1e-13)
    L,R,D=[state[k].double().numpy() for k in ['L','R','D']]
    shared=D@((R@mean)[:,None]*L+(L@mean)[:,None]*R)
    def f(z):return ((z@L.T)*(z@R.T))@D.T
    rows={(x['pair_id'],x['endpoint'],x['region'],x['predictor']):x for x in result['rows']};assert len(rows)==984
    checked=0;maxprediction=0.;maxmetric=0.;maxnative=0.
    for fi,file in enumerate(result['files']):
        assert sha(file['path'])==file['sha256'];s=torch.load(file['path'],weights_only=True)
        for z,w in zip(s['z'],s['writes']):
            fn=f(z.double().numpy());wn=w.double().numpy();error=float(np.linalg.norm(fn-wn)/max(np.linalg.norm(wn),1));assert error<1e-5;maxnative=max(maxnative,error)
        for entry in s['entries']:
            ep=entry['endpoint'];pos=entry['recipient_positions'];dp=entry['donor_positions'];zr=s['z'][ep].double().numpy()[:,pos];zd=s['z'][1-ep].double().numpy()[:,dp];dz=zd-zr;mid=(zr+zd)/2
            references=dict(shared_J_at_training_mean=dz@shared.T,recipient_J=f(zr+dz/2)-f(zr-dz/2),midpoint_J_exact_control=f(zd)-f(zr))
            nd=(s['writes'][1-ep].numpy()[:,dp]-s['writes'][ep].numpy()[:,pos]).astype(float)
            for name,value in entry['predictions'].items():
                pred=value.double().numpy();error=float(np.linalg.norm(pred-references[name])/max(np.linalg.norm(references[name]),1));assert error<1e-5;maxprediction=max(maxprediction,error)
                row=rows[s['pair_id'],ep,entry['region'],name]
                for field,val in [('target_delta_energy',float(np.square(nd).sum())),('predicted_delta_energy',float(np.square(pred).sum())),('delta_error',float(np.square(pred-nd).sum()))]:
                    assert np.isclose(val,row[field],rtol=1e-10,atol=1e-8);maxmetric=max(maxmetric,abs(val-row[field]))
                checked+=1
        if (fi+1)%8==0:print('LOCAL_AUDIT',fi+1,flush=True)
    assert checked==984 and len(result['ineligible'])==8
    out=dict(status='passed',result_sha256=sha(rp),source_sha256=sha(__file__),reference_mean_independent=True,checked_predictions=checked,max_prediction_relative_error=maxprediction,max_native_float64_relative_error=maxnative,max_metric_absolute_error=maxmetric,scope='All training reference indices/mean checked, shared Jacobian materialized explicitly once, recipient/midpoint predictions independently computed by NumPy quadratic polarization. All native endpoint writes and local energy metrics checked. No downstream evaluation yet.')
    path=V/'native_input_changes_audit.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
