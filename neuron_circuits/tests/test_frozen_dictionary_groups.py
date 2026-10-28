import unittest
import torch
from run_bilin18_frozen_dictionary_groups import greedy_union
from feature_structures.models import union_support
class GroupUnionTests(unittest.TestCase):
 def test_shared_atom_is_decoded_once(self):
  z=torch.tensor([[3.,4.,10.,1.]])
  M=torch.tensor([[1.,0.,1.,0.],[0.,1.,1.,1.]])
  q,g=greedy_union(z,M);self.assertEqual(g.sum().item(),2);torch.testing.assert_close(q,torch.ones_like(z));torch.testing.assert_close(z*q,z);torch.testing.assert_close(q,union_support(g,M))
 def test_second_gate_uses_uncovered_energy(self):
  z=torch.tensor([[10.,9.,8.,7.]])
  M=torch.tensor([[1.,1.,0.,0.],[1.,0.,1.,0.],[0.,0.,1.,1.]])
  q,g=greedy_union(z,M);self.assertEqual(g.nonzero()[:,1].tolist(),[0,2]);torch.testing.assert_close(q,torch.ones_like(z))
 def test_atom_permutation_preserves_decoded_result(self):
  z=torch.tensor([[10.,9.,8.,7.]])
  M=torch.tensor([[1.,1.,0.,0.],[1.,0.,1.,0.],[0.,0.,1.,1.]])
  perm=torch.tensor([2,0,3,1]);q,g=greedy_union(z,M);qp,gp=greedy_union(z[:,perm],M[:,perm]);torch.testing.assert_close(g,gp);torch.testing.assert_close((z*q)[:,perm],z[:,perm]*qp)
if __name__=='__main__':unittest.main()
