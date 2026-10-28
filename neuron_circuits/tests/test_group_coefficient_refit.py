import unittest
import torch
from run_bilin18_group_coefficient_refit import project_supports,best_refit_pair
class RefitTests(unittest.TestCase):
 def test_rank_deficient_support_matches_pseudoinverse(self):
  D=torch.tensor([[1.,0.,0.],[2.,0.,0.],[0.,1.,0.]]);q=torch.tensor([[1.,1.,0.],[1.,0.,1.]]);x=torch.tensor([[3.,4.,5.],[2.,3.,4.]]);b=torch.tensor([.1,.2,.3]);p=project_supports(x,D,b,q)
  for i in range(2):
   A=D[q[i].bool()].double();expected=b.double()+(x[i].double()-b.double())@torch.linalg.pinv(A)@A;torch.testing.assert_close(p[i],expected)
 def test_best_pair_matches_independent_lstsq(self):
  torch.manual_seed(729);D=torch.randn(5,6);x=torch.randn(4,6);b=torch.randn(6);S=torch.tensor([[1.,1.,1.,0.,0.],[0.,1.,1.,1.,0.],[1.,0.,0.,1.,1.]])
  q,e,orth=best_refit_pair(x,D,b,S,2)
  for n in range(4):
   errors=[]
   for s in S:
    A=D[s.bool()].double().T;y=x[n].double()-b.double();c=torch.linalg.lstsq(A,y).solution;errors.append((A@c-y).square().sum())
   errors=torch.stack(errors);torch.testing.assert_close(q[n],S[errors.argmin()]);torch.testing.assert_close(e[n].double(),errors.min(),atol=1e-5,rtol=1e-5)
if __name__=='__main__':unittest.main()
