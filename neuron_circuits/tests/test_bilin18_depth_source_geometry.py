import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_depth_source_geometry import fit_response_bases

def test_response_bases_distinguish_mean_from_centered_variation():
    x=torch.tensor([[10.,-3.,0.],[10.,-1.,0.],[10.,1.,0.],[10.,3.,0.]],dtype=torch.float64)
    bases,mean=fit_response_bases(x,2)
    torch.testing.assert_close(mean,torch.tensor([10.,0.,0.],dtype=torch.float64))
    assert abs(bases['global'][0,0])>.999
    assert abs(bases['centered_global'][0,1])>.999
    for name,b in bases.items():torch.testing.assert_close(b@b.T,torch.eye(len(b),dtype=b.dtype))

def test_output_tensor_gram_matches_explicit_symmetric_forms_and_gauge():
    from audit_bilin18_depth_tensor_output import output_tensor_gram
    torch.manual_seed(88);L=torch.randn(7,5,dtype=torch.float64);R=torch.randn(7,5,dtype=torch.float64);D=torch.randn(4,7,dtype=torch.float64)
    forms=.5*(torch.einsum('hi,hj->hij',L,R)+torch.einsum('hi,hj->hij',R,L))
    q=torch.einsum('oh,hij->oij',D,forms).flatten(1)
    torch.testing.assert_close(output_tensor_gram(L,R,D),q@q.T)
    a=torch.tensor([-.3,2.,1.,-4.,.7,5.,-2.],dtype=L.dtype);b=torch.arange(1,8,dtype=L.dtype)
    torch.testing.assert_close(output_tensor_gram(L*a[:,None],R*b[:,None],D/(a*b)[None]),q@q.T)
    torch.testing.assert_close(output_tensor_gram(R,L,D),q@q.T)
