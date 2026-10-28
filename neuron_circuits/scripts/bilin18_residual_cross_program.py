"""Exact centered low-rank residual polynomial; evaluation needs no native tensors."""
import torch

def compile_program(mean,core,partner,L,R,D,p):
    from run_bilin18_l17_residual_tensor_partners import residual_bilinear_matrix
    from run_bilin18_l17_residual_term_accounting import mean_linear
    d=len(mean);k=core.shape[1];ij=torch.triu_indices(k,k,device=core.device);mat=torch.stack([residual_bilinear_matrix(c,L,R,D,p).T for c in core.T]) # core,input,output
    within=torch.einsum('id,kdo->kio',core.T,mat)
    within=within[ij[0],ij[1]]*torch.where(ij[0]==ij[1],1.,2.)[:,None]
    cross=2*torch.einsum('dj,kdo->kjo',partner,mat)
    linear=mean_linear(torch.eye(d,device=mean.device,dtype=mean.dtype),mean,L,R,D,p)
    return {'mean':mean,'core':core,'partner':partner,'linear':linear,'within':within,'cross':cross}

def forward(x,p,include_cross=True):
    a=x-p['mean'];u=a@p['core'];ij=torch.triu_indices(u.shape[1],u.shape[1],device=x.device);out=a@p['linear']+(u[:,ij[0]]*u[:,ij[1]])@p['within']
    if include_cross:out=out+torch.einsum('ni,nj,ijo->no',u,a@p['partner'],p['cross'])
    return out

def counts(p):return {k:v.numel() for k,v in p.items()}

def change(x,z,p,include_cross=True):
    """Stable polynomial difference without subtracting large intact outputs."""
    delta=z-x;centered_sum=(x-p['mean'])+(z-p['mean']);du=delta@p['core'];su=centered_sum@p['core'];ij=torch.triu_indices(du.shape[1],du.shape[1],device=x.device)
    pairs=.5*(su[:,ij[0]]*du[:,ij[1]]+du[:,ij[0]]*su[:,ij[1]])
    out=delta@p['linear']+pairs@p['within']
    if include_cross:
        dv=delta@p['partner'];sv=centered_sum@p['partner'];out=out+.5*(torch.einsum('ni,nj,ijo->no',su,dv,p['cross'])+torch.einsum('ni,nj,ijo->no',du,sv,p['cross']))
    return out

def intact_forward(x,p):
    """Anchored residual correction; requires an explicit frozen anchor array."""
    return p['anchor']+forward(x,p)
