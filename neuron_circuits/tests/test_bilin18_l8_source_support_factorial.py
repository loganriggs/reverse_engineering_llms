import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l8_source_support_factorial import norm_match,changes


def test_sham_preserves_each_site_norm_including_zero_reference():
    torch.manual_seed(12)
    reference=torch.randn(20,9,dtype=torch.float64);reference[0]=0
    candidate=torch.randn_like(reference)
    torch.testing.assert_close(norm_match(candidate,reference).norm(dim=-1),reference.norm(dim=-1))


def test_batched_response_matches_scalar_differences():
    torch.manual_seed(13)
    q=torch.randn(4,9,9,dtype=torch.float64);q=(q+q.transpose(1,2))/2
    a=torch.randn_like(q);a=(a+a.transpose(1,2))/2
    x=torch.randn(20,9,dtype=torch.float64);y=torch.randn_like(x);mean=torch.randn(9,dtype=torch.float64)
    def scalar(z):
        c=z-mean
        return 2*torch.einsum('nd,adj,j->na',c,q,mean)+torch.einsum('nd,adj,nj->na',c,a,c)
    torch.testing.assert_close(changes(x,y,a,q,mean),scalar(y)-scalar(x))
    target=torch.einsum('nd,adj,nj->na',y,q,y)-torch.einsum('nd,adj,nj->na',x,q,x)
    torch.testing.assert_close(changes(x,y,q,q,mean),target)
