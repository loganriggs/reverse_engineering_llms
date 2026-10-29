"""Raw normalized-input to raw native-write adapter; no native write input."""
import torch
from torch import nn
from bilinear_input_featurizer import BilinearInputFeaturizer
class InputWritePredictor(nn.Module):
 def __init__(self,state):
  super().__init__();self.features=BilinearInputFeaturizer(state['kind'],state['seed']);self.features.load_state_dict(state['state'])
  for key,value in state['statistics'].items():self.register_buffer(key,value.clone())
 def forward(self,z):
  prediction,support,code=self.features((z-self.input_mean)/self.input_scale)
  return prediction*self.output_scale+self.output_mean,support,code
