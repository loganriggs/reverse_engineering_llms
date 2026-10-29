"""Executable M79 piecewise affine input predictor using frozen full chart matrices."""
import json
import numpy as np
import torch
from fit_pca_encoder import sha,V
class LocalAffineChartPredictor:
    def __init__(self,name):
        fp=V/'local_affine_chart_fit.json';fit=json.loads(fp.read_text());audit=json.loads((V/'local_affine_chart_fit_audit.json').read_text());assert audit['status']=='passed' and audit['fit_sha256']==sha(fp)
        entry=next(e for e in fit['models'] if e['name']==name);assert sha(entry['path'])==entry['sha256'];s=np.load(entry['path']);self.anchors=s['anchors'];self.slopes=s['slopes'];self.intercepts=s['intercepts'];self.rank=1152
    def routes(self,x):
        flat=x.reshape(-1,1152);dist=((flat*flat).sum(1)[:,None]+(self.anchors*self.anchors).sum(1)[None,:]-2*flat@self.anchors.T).clip(0)
        return dist.argmin(1),dist
    @torch.no_grad()
    def reconstruct(self,z,variant=None):
        x=z.detach().cpu().double().numpy();flat=x.reshape(-1,1152);labels,dist=self.routes(x);pred=np.empty_like(flat)
        for k in np.unique(labels):pred[labels==k]=flat[labels==k]@self.slopes[k]+self.intercepts[k]
        pred=pred.reshape(x.shape).astype('float32')
        return torch.from_numpy(pred).to(z.device),torch.ones(x.shape,device=z.device),torch.from_numpy(x).to(z.device),dict(rank_mean=1152.,numerical_nonzero_mean=float((x!=0).sum(-1).mean()))
    def __call__(self,z):return self.reconstruct(z)[0]
