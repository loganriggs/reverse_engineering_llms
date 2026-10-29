"""Parameter-matched continuous block alternative; compatible hard-block state dicts."""
import torch.nn.functional as F
from train_write_featurizers import Featurizer
from continuous_block_shrinkage import block_shrinkage


class ContinuousBlockFeaturizer(Featurizer):
    def __init__(self,seed):
        super().__init__('block',seed)

    def forward(self,x):
        raw=F.linear(x,self.encoder,self.encoder_bias)
        grouped=raw.reshape(*raw.shape[:-1],32,4)
        shrunk=block_shrinkage(grouped,8)
        code=shrunk.reshape_as(raw)
        support=(shrunk.norm(dim=-1)>0).unsqueeze(-1).expand_as(shrunk).reshape_as(raw).to(raw.dtype)
        return code@F.normalize(self.dictionary,dim=0).T+self.bias,support,code
