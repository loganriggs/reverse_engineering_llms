import numpy as np
import torch
from deep_multitask import DeepMultiTask

def load_model(file):
    state=torch.load(file,weights_only=True);A=state['A.0'];shared=A.ndim==2
    net=DeepMultiTask(A.shape[-1],A.shape[-2],state['W'].shape[0],len([k for k in state if k.startswith('A.')])+1,shared,0);net.load_state_dict(state);return net

def features(net,x):
    z=net.hidden(x);i,j=net.i,net.j
    return z[:,i]*z[:,j] if net.shared else (z[:,:12,i]*z[:,:12,j]).reshape(len(x),-1)

def numpy_features(state,x):
    shared=state['A.0'].ndim==2;z=x if shared else np.repeat(x[:,None,:],state['W'].shape[0],axis=1)
    depth=len([k for k in state if k.startswith('A.')])
    for l in range(depth):
        A,B,scale=[state[f'{k}.{l}'] if k in ['A','B'] else state[f'scale_{l}'] for k in ['A','B','scale']]
        z=((z@A.T)*(z@B.T) if shared else np.einsum('ntd,thd->nth',z,A)*np.einsum('ntd,thd->nth',z,B))/scale
    i,j=np.triu_indices(z.shape[-1]);phi=z[:,i]*z[:,j] if shared else (z[:,:12,i]*z[:,:12,j]).reshape(len(x),-1)
    pred=phi@state['W'].T if shared else np.einsum('ntp,tp->nt',z[:,:,i]*z[:,:,j],state['W'])
    return phi,pred
