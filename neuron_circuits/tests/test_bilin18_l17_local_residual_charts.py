import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_local_residual_charts import local_chart,local_project

def test_local_chart_orthogonality_and_projection():
    torch.manual_seed(98);state=torch.randn(4,9,dtype=torch.float64);w=torch.randn(17,9,dtype=torch.float64)
    q,s,mass,ids=local_chart(state,w,12,5)
    torch.testing.assert_close(q@q.transpose(1,2),torch.eye(5,dtype=state.dtype).expand(4,5,5))
    x=torch.randn_like(state);px=local_project(x,q);torch.testing.assert_close(local_project(px,q),px)
    assert torch.all(px.square().sum(-1)<=x.square().sum(-1)+1e-10)
    assert torch.all((mass>0)&(mass<=1)) and ids.shape==(4,12)
