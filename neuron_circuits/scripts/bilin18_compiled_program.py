"""Standalone fixed quadratic inference; requires only numpy and torch."""
from pathlib import Path
import numpy as np
import torch

def load_program(path,device='cpu',dtype=torch.float32):
    with np.load(Path(path)) as data:return {k:torch.as_tensor(data[k].copy(),device=device,dtype=dtype) for k in data.files}

def factorize_program(program,max_rank=32):
    eigenvalues=[];eigenvectors=[];errors=[]
    for form in program['forms']:
        w,v=torch.linalg.eigh(form.double());ids=torch.argsort(w.abs(),descending=True)[:min(max_rank,len(w))];w=w[ids];v=v[:,ids]
        errors.append(float((form.double()-(v*w[None])@v.T).norm()/form.double().norm()))
        eigenvalues.append(w.to(form.dtype));eigenvectors.append(v.to(form.dtype))
    result={k:v for k,v in program.items() if k!='forms'};result['eigenvalues']=torch.stack(eigenvalues);result['eigenvectors']=torch.stack(eigenvectors)
    return result,errors

def forward_normalized(x,program):
    z=x@program['reads']-program['offset']
    if 'forms' in program:quadratic=torch.einsum('ni,oij,nj->no',z,program['forms'],z)
    else:
        modes=torch.einsum('nk,okr->nor',z,program['eigenvectors'])
        quadratic=(modes.square()*program['eigenvalues'][None]).sum(-1)
    coeff=quadratic+z@program['linear'].T
    return coeff@program['outputs']+program['bias']

def forward_raw(raw,program):
    normalized=raw/(raw.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
    return forward_normalized(normalized,program)
