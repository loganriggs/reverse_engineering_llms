"""M79 training-only anchors and explicit native Taylor charts."""
import json
import numpy as np
import torch
from fit_pca_encoder import sha,V,A

def nearest(z,c):
    dist=((z*z).sum(1)[:,None]+(c*c).sum(1)[None,:]-2*z@c.T).clip(0)
    return dist.argmin(1),dist

def main():
    torch.set_num_threads(4)
    pp=V/'local_affine_chart_protocol.json';p=json.loads(pp.read_text())
    for name,digest in p['references'].items():assert sha(V/name)==digest
    toy=json.loads((V/'local_affine_chart_toys.json').read_text());assert toy['status']=='passed' and toy['protocol_sha256']==sha(pp)
    rp=V/'native_input_reference.json';ref=json.loads(rp.read_text());assert sha(ref['path'])==ref['sha256'];s=torch.load(ref['path'],weights_only=True);z=s['z'][s['indices']].double().numpy();assert len(z)==4092
    rng=np.random.default_rng(7901);selected=rng.choice(len(z),32,replace=False);random=z[selected].copy();c=random.copy();history=[]
    for step in range(50):
        labels,dist=nearest(z,c);history.append(float(dist[np.arange(len(z)),labels].sum()))
        for k in range(32):
            if np.any(labels==k):c[k]=z[labels==k].mean(0)
    assert np.all(np.diff(history)<=1e-7)
    L,R,D=[s[k].double() for k in ['L','R','D']];models=[]
    for name,centers in [('mean',z.mean(0,keepdims=True)),('random32',random),('kmeans32',c)]:
        slopes=[];intercepts=[]
        for i,center in enumerate(centers):
            ct=torch.from_numpy(center);lc=L@ct;rc=R@ct
            slope=(L.T*rc)@D.T+(R.T*lc)@D.T
            intercept=-(lc*rc)@D.T
            slopes.append(slope.numpy());intercepts.append(intercept.numpy())
        labels,dist=nearest(z,centers);state=A/f'emlp_m79_charts_{name}.npz';assert not state.exists()
        np.savez(state,anchors=centers,slopes=np.stack(slopes),intercepts=np.stack(intercepts),training_indices=s['indices'].numpy(),random_initial_indices=selected)
        entry=dict(name=name,path=str(state),sha256=sha(state),charts=len(centers),coefficients=len(centers)*(1152**2+1152),training_distance_mean=float(dist[np.arange(len(z)),labels].mean()),training_occupancy=np.bincount(labels,minlength=len(centers)).tolist());models.append(entry);print(name,entry['training_distance_mean'],flush=True)
    out=dict(status='frozen',source_sha256=sha(__file__),protocol_sha256=sha(pp),reference_sha256=sha(rp),models=models,kmeans_objective_history=history,training_positions=4092,evaluation_positions_used=0,scope='Anchors and full explicit chart coefficients frozen before evaluation; all native weights reused, no compression claim.')
    dest=V/'local_affine_chart_fit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
