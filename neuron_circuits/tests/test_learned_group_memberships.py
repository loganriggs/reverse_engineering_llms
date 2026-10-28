import unittest
import itertools
import torch
from run_bilin18_learned_group_memberships import select_pairs,union_pairs,update_memberships
class MembershipTests(unittest.TestCase):
 def test_coordinate_update_monotone_and_anchored(self):
  torch.manual_seed(730);w=torch.rand(100,8).square();M=torch.eye(8).scatter_(1,torch.rand(8,8).fill_diagonal_(-1).topk(3,dim=1).indices,1);pairs=select_pairs(w,M);before=(w*union_pairs(M,pairs)).sum();new=update_memberships(w,M,pairs);self.assertGreaterEqual(float((w*union_pairs(new,pairs)).sum()),float(before)-1e-5);self.assertTrue(bool((new.sum(-1)==4).all()));torch.testing.assert_close(new.diag(),torch.ones(8));self.assertTrue(bool((pairs[:,0]!=pairs[:,1]).all()))
 def test_assignments_do_not_reduce_greedy_energy(self):
  torch.manual_seed(731);w=torch.rand(100,8);M=torch.eye(8).scatter_(1,torch.rand(8,8).fill_diagonal_(-1).topk(3,dim=1).indices,1);a=select_pairs(w,M,0);b=select_pairs(w,M,4);self.assertTrue(bool(((w*union_pairs(M,b)).sum(-1)>=(w*union_pairs(M,a)).sum(-1)-1e-6).all()))
 def test_updates_match_exhaustive_membership_objective(self):
  torch.manual_seed(732);w=torch.rand(20,6,dtype=torch.double);M=torch.eye(6,dtype=torch.double).scatter_(1,torch.rand(6,6).fill_diagonal_(-1).topk(3,dim=1).indices,1);pairs=select_pairs(w,M);expected=M.clone()
  for g in range(6):
   if not (pairs==g).any():continue
   candidates=[];scores=[]
   for neighbors in itertools.combinations([i for i in range(6) if i!=g],3):
    candidate=expected.clone();candidate[g].zero_();candidate[g,[g,*neighbors]]=1;candidates.append(candidate);scores.append(float((w*union_pairs(candidate,pairs)).sum()))
   expected=candidates[max(range(len(scores)),key=scores.__getitem__)]
  actual=update_memberships(w,M,pairs);torch.testing.assert_close((w*union_pairs(actual,pairs)).sum(),(w*union_pairs(expected,pairs)).sum())
 def test_shared_members_count_once(self):
  M=torch.tensor([[1.,1.,0.],[0.,1.,1.]]);torch.testing.assert_close(union_pairs(M,torch.tensor([[0,1]])),torch.ones(1,3))
if __name__=='__main__':unittest.main()
