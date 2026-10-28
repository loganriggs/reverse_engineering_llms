import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l8_head3_payload_training import quota_indices,payload_panel

def test_fixed_source_quota_and_even_control_excludes_supplement():
    torch.manual_seed(3)
    ids=quota_indices(101,42,True,'cpu')
    assert torch.bincount(ids//101,minlength=6).tolist()==[42]*6
    ids=quota_indices(101,42,False,'cpu')
    assert len(ids)==252 and ids.max()<505
    assert all(torch.bincount(ids[:210]//101,minlength=5)==42)

def test_payload_actual_replay_and_sham_norm_audit():
    torch.manual_seed(3)
    pre=torch.randn(17,12);heads=torch.randn(17,9,12);projection=torch.randn(12,4)
    p,error=payload_panel(pre,heads,projection,99)
    torch.testing.assert_close(p['actual'],torch.nn.functional.rms_norm(pre-heads[:,3],(12,)))
    assert error<1e-6
    for x in p.values():assert torch.isfinite(x).all()
