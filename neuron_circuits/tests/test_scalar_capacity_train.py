import unittest
import torch
from run_bilin18_scalar_capacity_train import SignedTopK
from run_bilin18_verified_head_sparse import SignedBSF
class CapacityTests(unittest.TestCase):
 def test_topk_preserves_signed_largest_coefficients(self):
  m=SignedTopK(2)
  with torch.no_grad():m.encoder.weight.copy_(torch.eye(128));m.encoder.bias.zero_()
  x=torch.zeros(1,128);x[0,:4]=torch.tensor([-9.,4.,-3.,2.]);z,mask=m.encode(x);self.assertEqual(mask.sum().item(),2);torch.testing.assert_close(z[0,:4],torch.tensor([-9.,4.,0.,0.]))
 def test_k8_identical_to_reference_and_full_budget_dense(self):
  torch.manual_seed(735);m=SignedTopK(8);old=SignedBSF(1);old.load_state_dict(m.state_dict());x=torch.randn(4,128);torch.testing.assert_close(m(x),old(x),rtol=0,atol=0);dense=SignedTopK(128);dense.load_state_dict(m.state_dict());torch.testing.assert_close(dense(x),dense.encoder(x)@dense.decoder+dense.bias)
 def test_parameter_count_constant(self):
  self.assertEqual({sum(p.numel() for p in SignedTopK(k).parameters()) for k in [8,16,32,64]}, {33024})
if __name__=='__main__':unittest.main()
