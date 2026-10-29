"""M111 checkpoint compatibility and full adapter forward/backward validation."""
import json
import torch
from fit_pca_encoder import V,sha
from train_write_featurizers import Featurizer
from continuous_write_featurizer import ContinuousBlockFeaturizer
from continuous_block_shrinkage import block_shrinkage


def main():
    torch.set_num_threads(1);torch.manual_seed(11102)
    checks=[]
    for seed in [71,72]:
      hard=Featurizer('block',seed).double();soft=ContinuousBlockFeaturizer(seed).double();soft.load_state_dict(hard.state_dict(),strict=True)
      assert set(hard.state_dict())==set(soft.state_dict())
      assert sum(p.numel() for p in hard.parameters())==sum(p.numel() for p in soft.parameters())
      for shape in [(1152,),(5,1152),(2,3,1152)]:
        x=torch.randn(shape,dtype=torch.float64,requires_grad=True);rec,support,code=soft(x)
        raw=torch.nn.functional.linear(x,soft.encoder,soft.encoder_bias);expected=block_shrinkage(raw.reshape(*raw.shape[:-1],32,4),8).reshape_as(raw)
        assert torch.equal(code,expected) and bool((support.sum(-1)<=32).all())
        rec.square().mean().backward()
        assert torch.isfinite(x.grad).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in soft.parameters())
        checks.append(dict(seed=seed,input_shape=list(shape),active_min=int(support.sum(-1).min()),active_max=int(support.sum(-1).max()),parameter_count=sum(p.numel() for p in soft.parameters())))
        soft.zero_grad()
    dest=V/'continuous_featurizer_adapter_checks.json';assert not dest.exists();dest.write_text(json.dumps(dict(status='passed',source_sha256=sha(__file__),adapter_sha256=sha(V.parent/'continuous_write_featurizer.py'),map_sha256=sha(V.parent/'continuous_block_shrinkage.py'),checks=checks),indent=2)+'\n');print('ADAPTER_CHECKS',len(checks),'PARAMETERS',checks[0]['parameter_count'])
if __name__=='__main__':main()
