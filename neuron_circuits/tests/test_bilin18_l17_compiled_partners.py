import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_compiled_partners import compile_partners,coefficients,normalized_forward
from run_bilin18_l8_context_role_compression import state_rules

def test_compiled_union_reproduces_projected_cross_state_and_finite_change():
    torch.manual_seed(94);d=torch.float64
    L=torch.randn(23,19,dtype=d);R=torch.randn(23,19,dtype=d);D=torch.randn(11,23,dtype=d)
    p=torch.linalg.qr(torch.randn(19,2,dtype=d)).Q;mu=torch.randn(19,dtype=d);u=torch.linalg.qr(torch.randn(11,3,dtype=d)).Q.T
    prog,_=compile_partners(L,R,D,p,mu,u);assert prog['reads'].shape[1]<=2+3*2+3
    x=torch.randn(8,19,dtype=d);y=torch.randn_like(x)
    def ref(z):return state_rules(z,mu,p.T,p,L,R,D)['cross']@u.T
    torch.testing.assert_close(coefficients(x,prog),ref(x),atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(normalized_forward(y,prog)-normalized_forward(x,prog),(ref(y)-ref(x))@u,atol=1e-9,rtol=1e-9)

def test_spectral_program_matches_dense_program():
    from bilin18_compiled_program import factorize_program,forward_normalized
    torch.manual_seed(95);d=torch.float64
    L=torch.randn(31,25,dtype=d);R=torch.randn(31,25,dtype=d);D=torch.randn(9,31,dtype=d)
    p=torch.linalg.qr(torch.randn(25,3,dtype=d)).Q;mu=torch.randn(25,dtype=d);u=torch.linalg.qr(torch.randn(9,2,dtype=d)).Q.T
    dense,_=compile_partners(L,R,D,p,mu,u);spectral,errors=factorize_program(dense,6)
    assert max(errors)<1e-12
    x=torch.randn(12,25,dtype=d)
    torch.testing.assert_close(forward_normalized(x,dense),forward_normalized(x,spectral),atol=1e-9,rtol=1e-9)
