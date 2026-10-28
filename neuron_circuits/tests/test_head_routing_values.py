import unittest
import torch
from run_bilin18_head_routing_values import factor_deltas
class RoutingValueTests(unittest.TestCase):
 def test_signed_nonconvex_factorization(self):
  torch.manual_seed(738);a,b=torch.randn(2,7,dtype=torch.double);Lr,Ld,Br,Bd=torch.randn(4,7,5,dtype=torch.double);lam=4.625;d=factor_deltas(a,b,Lr,Ld,Br,Bd,lam);expected=torch.zeros(5,dtype=torch.double)
  for t in range(7):expected+=b[t]*((1-lam)*Ld[t]+lam*Bd[t])-a[t]*((1-lam)*Lr[t]+lam*Br[t])
  torch.testing.assert_close(d['full'],expected);torch.testing.assert_close(d['routing']+d['values']+d['interaction'],expected);torch.testing.assert_close(d['local']+d['bus'],d['values'])
 def test_fixed_factor_limits(self):
  torch.manual_seed(739);a,b=torch.randn(2,4,dtype=torch.double);Lr,Ld,Br,Bd=torch.randn(4,4,3,dtype=torch.double);d=factor_deltas(a,a,Lr,Ld,Br,Bd,-2.);torch.testing.assert_close(d['full'],d['values']);torch.testing.assert_close(d['routing'],torch.zeros(3,dtype=torch.double));d=factor_deltas(a,b,Lr,Lr,Br,Br,2.);torch.testing.assert_close(d['full'],d['routing']);torch.testing.assert_close(d['interaction'],torch.zeros(3,dtype=torch.double))
if __name__=='__main__':unittest.main()
