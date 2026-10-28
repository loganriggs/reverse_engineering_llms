import unittest
import torch
from run_bilin18_reconstruction_group_gates import best_support
class ExactGateTests(unittest.TestCase):
 def test_enumeration_matches_independent_nonorthogonal_bruteforce(self):
  torch.manual_seed(728);x=torch.randn(7,5,dtype=torch.double);z=torch.randn(7,4,dtype=torch.double);D=torch.randn(4,5,dtype=torch.double);b=torch.randn(5,dtype=torch.double);M=torch.tensor([[1,1,0,0],[0,1,1,0],[0,0,1,1]],dtype=torch.double);pairs=[(0,1),(0,2),(1,2)];S=torch.stack([((M[i]+M[j])>0).double() for i,j in pairs]);q,e=best_support(x,z,D,b,S,chunk=2)
  for n in range(len(x)):
   errors=[]
   for s in S:
    pred=b.clone()
    for a in range(4):
     if s[a]:pred+=z[n,a]*D[a]
    errors.append((pred-x[n]).square().sum())
   errors=torch.stack(errors);torch.testing.assert_close(e[n],errors.min());torch.testing.assert_close(q[n],S[errors.argmin()])
 def test_bias_and_shared_atoms_count_once(self):
  D=torch.eye(3);z=torch.tensor([[2.,3.,4.]]);b=torch.ones(3);S=torch.tensor([[1.,1.,1.],[1.,0.,1.]]);q,e=best_support(z+b,z,D,b,S,1);torch.testing.assert_close(q,S[:1]);self.assertEqual(e.item(),0.)
if __name__=='__main__':unittest.main()
