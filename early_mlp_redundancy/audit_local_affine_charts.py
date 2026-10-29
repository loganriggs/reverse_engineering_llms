"""M79 independent direct-distance routing and native-polynomial chart audit."""
import json
import numpy as np
import torch
from fit_pca_encoder import sha,V

def route(z,c):
    return np.concatenate([np.sum((b[:,None,:]-c[None,:,:])**2,axis=2).argmin(1) for b in np.array_split(z,32)])

def main():
    torch.set_num_threads(1);fp=V/'local_affine_chart_fit.json';fit=json.loads(fp.read_text());ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];s=torch.load(ref['path'],weights_only=True);z=s['z'][s['indices']].numpy().astype('float64');L,R,D=[s[k].numpy().astype('float64') for k in ['L','R','D']]
    def f(x):return ((x@L.T)*(x@R.T))@D.T
    selected=np.random.default_rng(7901).choice(len(z),32,replace=False);c=z[selected].copy()
    for step in range(50):
        labels=route(z,c)
        c=np.stack([np.sum(z[labels==i],axis=0)/np.sum(labels==i) if np.any(labels==i) else c[i] for i in range(32)])
    expected={'mean':np.sum(z,axis=0,keepdims=True)/len(z),'random32':z[selected],'kmeans32':c};rows=[]
    for e in fit['models']:
        assert sha(e['path'])==e['sha256'];state=np.load(e['path']);centers=state['anchors'];assert np.allclose(centers,expected[e['name']],rtol=1e-12,atol=1e-12)
        labels=route(z,centers);assert np.bincount(labels,minlength=len(centers)).tolist()==e['training_occupancy']
        maxerr=0.;maxrem=0.
        for k,center in enumerate(centers):
            # Deterministic dispersed training points; every stored chart checked.
            ix=(np.arange(8)*509+k*13)%len(z);x=z[ix];delta=x-center
            analytic=f(center)+((center@L.T)*(delta@R.T)+(delta@L.T)*(center@R.T))@D.T
            explicit=x@state['slopes'][k]+state['intercepts'][k]
            err=np.linalg.norm(explicit-analytic)/max(1,np.linalg.norm(analytic));rem=np.linalg.norm(f(x)-explicit-f(delta))/max(1,np.linalg.norm(f(x)))
            assert err<1e-8 and rem<1e-8;maxerr=max(maxerr,err);maxrem=max(maxrem,rem)
        rows.append(dict(name=e['name'],charts_checked=len(centers),routing_checks=len(z),max_prediction_relative=maxerr,max_remainder_relative=maxrem))
    out=dict(status='passed',source_sha256=sha(__file__),fit_sha256=sha(fp),models=rows,scope='Independent direct distances and centroid reconstruction for all50steps; all4092training routes per model; every explicit chart checked against factored native derivative and exact quadratic remainder on8training inputs. No evaluation inputs used.')
    p=V/'local_affine_chart_fit_audit.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
