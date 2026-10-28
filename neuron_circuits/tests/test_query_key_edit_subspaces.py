import unittest
import torch
from run_bilin18_query_key_edit_subspaces import rotate,rotary_values,projected_edit
class EditSubspaceTests(unittest.TestCase):
 def test_rotary_matches_native_formula_and_inverse(self):
  torch.manual_seed(742);x=torch.randn(128,dtype=torch.double)
  for pos in [0,7,23,127]:
   c,s=rotary_values(pos);a,b=x.chunk(2);expected=torch.cat([a*c+b*s,-a*s+b*c]);torch.testing.assert_close(rotate(x,pos),expected);torch.testing.assert_close(rotate(expected,pos,True),x)
 def test_projection_preserves_orthogonal_receiver_and_controls_norm(self):
  r=torch.tensor([3.,4.,0.],dtype=torch.double);d=torch.tensor([8.,2.,1.],dtype=torch.double);B=torch.eye(3,dtype=torch.double)[:,:1];x=projected_edit(r,d,B);torch.testing.assert_close(x,torch.tensor([8.,4.,0.],dtype=torch.double));torch.testing.assert_close(projected_edit(r,d,B,True).norm(),r.norm());torch.testing.assert_close(projected_edit(r,d,torch.eye(3,dtype=torch.double)),d)
if __name__=='__main__':unittest.main()
