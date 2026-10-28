import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l8_execution_alignment import shifted_indices
from run_bilin18_l8_source_support_factorial import norm_match

def test_shuffle_changes_row_preserves_position_and_recipient_norm():
    rows,positions=7,4
    ids=shifted_indices(rows,positions,3,'cpu');original=torch.arange(rows*positions)
    assert torch.equal(ids%positions,original%positions)
    assert torch.all(ids//positions!=original//positions)
    assert torch.equal(ids.sort().values,original)
    torch.manual_seed(1);payload=torch.randn(rows*positions,9)
    torch.testing.assert_close(norm_match(payload[ids],payload).norm(dim=-1),payload.norm(dim=-1))


def test_angle_control_preserves_norm_inner_product_and_rms_denominator():
    from run_bilin18_l8_execution_alignment import angle_matched_payload
    torch.manual_seed(42)
    q=torch.linalg.qr(torch.randn(17,6,dtype=torch.float64)).Q
    pre=torch.randn(12,17,dtype=torch.float64)
    ref=torch.randn(12,6,dtype=torch.float64)@q.T
    sham=angle_matched_payload(pre,ref,q,torch.randn(12,6,dtype=torch.float64))
    torch.testing.assert_close(sham.norm(dim=-1),ref.norm(dim=-1))
    torch.testing.assert_close((pre*sham).sum(-1),(pre*ref).sum(-1))
    torch.testing.assert_close((sham@q)@q.T,sham)
    for scale in [.001,.05,1.,2.]:
        torch.testing.assert_close((pre-scale*sham).norm(dim=-1),(pre-scale*ref).norm(dim=-1))
    assert not torch.allclose(sham,ref)
