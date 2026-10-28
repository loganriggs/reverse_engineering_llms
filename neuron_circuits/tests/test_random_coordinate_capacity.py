import unittest,itertools
import torch
from run_bilin18_random_coordinate_capacity import orthogonal_basis,reconstruct_coordinates
class RandomCoordinateTests(unittest.TestCase):
 def test_topk_is_best_orthogonal_support(self):
  Q=orthogonal_basis(737,6);p=torch.tensor([[1.,-2.,3.,.4,-.2,.8]],dtype=torch.double);r,m=reconstruct_coordinates(p,Q,3);z=p@Q;errs=[]
  for ids in itertools.combinations(range(6),3):
   c=torch.zeros_like(z);c[:,list(ids)]=z[:,list(ids)];errs.append(((c@Q.T-p)**2).sum())
  torch.testing.assert_close(((r-p)**2).sum(),min(errs));self.assertEqual(m.sum().item(),3)
 def test_seed_reproducibility_and_full_roundtrip(self):
  Q=orthogonal_basis(738,8);torch.testing.assert_close(Q,orthogonal_basis(738,8),rtol=0,atol=0);torch.testing.assert_close(Q.T@Q,torch.eye(8,dtype=torch.double));p=torch.randn(4,8,dtype=torch.double);r,_=reconstruct_coordinates(p,Q,8);torch.testing.assert_close(r,p)
if __name__=='__main__':unittest.main()
