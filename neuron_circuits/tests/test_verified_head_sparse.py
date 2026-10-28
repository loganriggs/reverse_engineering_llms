"""Independent small-case checks for the H724 signed sparse baselines."""
import unittest
import torch
from run_bilin18_verified_head_sparse import SignedBSF
class SparseHeadTests(unittest.TestCase):
 def identity(self,r):
  m=SignedBSF(r)
  with torch.no_grad():m.encoder.weight.copy_(torch.eye(128));m.encoder.bias.zero_();m.decoder.copy_(torch.eye(128));m.bias.zero_()
  return m
 def test_signed_scalar_topk_retains_negative_large_coordinates(self):
  m=self.identity(1);x=torch.arange(1,129,dtype=torch.float32)[None,:];x[:,::2]*=-1;expected=torch.zeros_like(x);expected[:,-8:]=x[:,-8:];torch.testing.assert_close(m(x),expected)
 def test_block_gate_selects_joint_energy_not_individual_axes(self):
  m=self.identity(4);x=torch.zeros(1,128);x[0,:4]=3;x[0,4:8]=2;x[0,8]=5;expected=torch.zeros_like(x);expected[0,:4]=3;expected[0,8]=5;torch.testing.assert_close(m(x),expected);_,mask=m.encode(x);self.assertEqual(mask.nonzero()[:,1].tolist(),[0,2])
 def test_signed_reconstruction_gradient_is_finite(self):
  m=self.identity(1);x=torch.arange(1,129,dtype=torch.float32)[None,:]/128;loss=(m(x)-2*x).square().mean();loss.backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
 def test_whiten_rotate_unwhiten_is_exact_without_truncation(self):
  torch.manual_seed(724);Q=torch.linalg.qr(torch.randn(128,128,dtype=torch.float64)).Q;sd=torch.linspace(.3,3,128,dtype=torch.float64);x=torch.randn(4,128,dtype=torch.float64);z=(x/sd)@Q;decoder=sd[:,None]*Q;torch.testing.assert_close(z@decoder.T,x,rtol=1e-12,atol=1e-12)
if __name__=='__main__':unittest.main()
