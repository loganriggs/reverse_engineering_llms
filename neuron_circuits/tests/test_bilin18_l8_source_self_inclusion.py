import sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l8_source_self_inclusion import baseline_and_edit
from run_bilin18_l8_execution_alignment import angle_matched_payload

def test_removed_baseline_reverses_actual_secant_and_preserves_sham_rms():
    torch.manual_seed(73);dtype=torch.float64
    q=torch.linalg.qr(torch.randn(13,5,dtype=dtype)).Q
    pre=torch.randn(9,13,dtype=dtype);ref=torch.randn(9,5,dtype=dtype)@q.T
    removed=baseline_and_edit(pre,ref,'removed');torch.testing.assert_close(removed+ref,pre)
    left=torch.randn(13,7,dtype=dtype);right=torch.randn(13,7,dtype=dtype);down=torch.randn(7,13,dtype=dtype)
    def f(z):
        x=F.rms_norm(z,(13,));return ((x@left)*(x@right))@down
    torch.testing.assert_close(f(removed+ref)-f(removed),-(f(pre-ref)-f(pre)))
    for z in [pre,removed]:
        p=angle_matched_payload(z,ref,q,torch.randn(9,5,dtype=dtype))
        for a in [-1.,-.05,.05,1.]:torch.testing.assert_close((z+a*p).norm(dim=-1),(z+a*ref).norm(dim=-1))

def test_exact_nonorthogonal_source_response_decomposition():
    from analyze_bilin18_l8_source_terms import addition_terms
    from run_bilin18_l8_context_role_compression import B
    torch.manual_seed(74);dtype=torch.float64
    c=torch.randn(11,13,dtype=dtype);h=torch.randn(11,13,dtype=dtype)
    L=torch.randn(19,13,dtype=dtype);R=torch.randn(19,13,dtype=dtype);D=torch.randn(13,19,dtype=dtype)
    terms=addition_terms(c,h,L,R,D);x=F.rms_norm(c,(13,));y=F.rms_norm(c+h,(13,));target=B(y,y,L,R,D)-B(x,x,L,R,D)
    torch.testing.assert_close(sum(terms.values()),target)
    credit=sum((t*target).sum() for t in terms.values())/target.square().sum()
    torch.testing.assert_close(credit,torch.ones_like(credit))
