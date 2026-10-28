import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_residual_tensor_partners import residual_bilinear_matrix
from bilin18_compiled_program import forward_normalized
from run_bilin18_l8_context_role_compression import B

def test_residual_bilinear_contraction_cancels_offsets_and_linear_terms():
    torch.manual_seed(125);c=torch.randn(7,dtype=torch.float64);v=torch.randn(7,dtype=torch.float64);L=torch.randn(11,7,dtype=torch.float64);R=torch.randn_like(L);D=torch.randn(7,11,dtype=torch.float64)
    p={k:torch.randn(*shape,dtype=torch.float64) for k,shape in {'reads':(7,4),'offset':(4,),'eigenvectors':(3,4,2),'eigenvalues':(3,2),'linear':(3,4),'outputs':(3,7),'bias':(7,)}.items()}
    def e(x):return B(x[None],x[None],L,R,D)[0]-forward_normalized(x[None],p)[0]
    expected=(e(c+v)-e(c)-e(v)+e(torch.zeros_like(c)))/2
    torch.testing.assert_close(residual_bilinear_matrix(c,L,R,D,p)@v,expected)
    torch.testing.assert_close(residual_bilinear_matrix(v,L,R,D,p)@c,expected)
