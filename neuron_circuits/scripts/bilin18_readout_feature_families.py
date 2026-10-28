"""H282 frozen unsupervised feature maps, fitted only on discovery training states."""
import torch

def features(x,p,m):
    z=(x-p['mean'])@m['reads']
    if 'pairs' in m:
        z=(z-m['read_mean'])/m['read_std']; ij=m['pairs'].long(); z=z[:,ij[0]]*z[:,ij[1]]
    z=(z-m['raw_mean'])@m['projection']
    return (z-m['feature_mean'])/m['feature_std']

def fit_features(x,p,family,cut):
    reads=torch.cat([p['core'],p['partner']],1); q=torch.linalg.qr(reads.double()).Q
    if family=='quadratic':
        z=(x-p['mean'])@reads; mean=z[:cut].mean(0); std=z[:cut].std(0,unbiased=False).clamp_min(1e-6);z=(z-mean)/std;ij=torch.triu_indices(64,64,device=x.device);raw=z[:,ij[0]]*z[:,ij[1]];m={'reads':reads,'read_mean':mean,'read_std':std,'pairs':ij}
    elif family=='outside':
        # All candidate directions are orthogonal to the original read span.
        eye=torch.eye(x.shape[1],device=x.device,dtype=torch.float64);reads=(eye-q@q.T).float();raw=(x-p['mean'])@reads;m={'reads':reads}
    else:raise ValueError(family)
    center=raw[:cut].mean(0);train=(raw[:cut]-center).double();_,v=torch.linalg.eigh(train.T@train);proj=v[:,-64:].flip(1).float();z=(raw-center)@proj;m.update(raw_mean=center,projection=proj,feature_mean=z[:cut].mean(0),feature_std=z[:cut].std(0,unbiased=False).clamp_min(1e-6));return features(x,p,m),m
