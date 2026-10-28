import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from feature_structures.models import *

class TestStructures(unittest.TestCase):
 def test_union_counts_shared_atom_once(self):
  M=torch.tensor([[1.,1.,1.,0.,0.],[0.,0.,1.,1.,1.]])
  s=torch.tensor([[1.,1.],[1.,0.],[0.,0.]])
  self.assertTrue(torch.equal(union_support(s,M),torch.tensor([[1.,1.,1.,1.,1.],[1.,1.,1.,0.,0.],[0.,0.,0.,0.,0.]])))
  m=SharedGroups(5,5,2,2,3)
  with torch.no_grad():m.dictionary.copy_(torch.eye(5))
  a=torch.tensor([[1.,2.,3.,4.,5.]])
  out=m(torch.zeros(1,5),gates=s[:1],coordinates=a,membership=M)
  torch.testing.assert_close(out['reconstruction'],a)
  self.assertEqual(float(out['costs']['coordinates']),5.)
  self.assertEqual(float(out['costs']['memberships']),6.)
  zero=m(torch.zeros(1,5),gates=s[:1],coordinates=torch.zeros_like(a),membership=M)
  self.assertEqual(float(zero['costs']['coordinates']),0.)
 def test_shared_atom_removal_specificity(self):
  M=torch.tensor([[1.,1.,0.],[0.,1.,1.],[1.,0.,0.]])
  m=SharedGroups(3,3,3,1,2)
  with torch.no_grad():m.dictionary.copy_(torch.eye(3))
  kw=dict(gates=torch.eye(3),coordinates=torch.ones(3,3),membership=M)
  a=m(torch.zeros(3,3),**kw)['reconstruction'];b=m(torch.zeros(3,3),remove_atoms=(1,),**kw)['reconstruction']
  torch.testing.assert_close(a-b,torch.tensor([[0.,1.,0.],[0.,1.,0.],[0.,0.,0.]]))
 def test_taxonomy_multiple_parents_and_no_double_decode(self):
  parents=[(),(),(0,1),(2,)]
  seeds=torch.tensor([[0.,0.,0.,1.]])
  torch.testing.assert_close(implication_closure(seeds,parents),torch.ones(1,4))
  torch.testing.assert_close(union_support(implication_closure(seeds,parents),torch.ones(4,1)),torch.ones(1,1))
 def test_cycle_and_bad_edges_rejected(self):
  for graph in [[(1,),()],[(0,)],[(),(0,0)]]:
   with self.assertRaises(ValueError):validate_parents(graph)
  with self.assertRaises(ValueError):ComputationDAG(1,1,[(),(0,)],products=[(),((0,1),)])
 def test_additive_refinements_are_additive(self):
  m=AdditiveRefinements(2,1,[(),(0,)])
  with torch.no_grad():m.dictionary.copy_(torch.tensor([[[1.],[0.]],[[0.],[1.]]]))
  y,g=m(torch.tensor([[[2.],[3.]]]),torch.tensor([[0.,1.]]))
  torch.testing.assert_close(y,torch.tensor([[2.,3.]]));torch.testing.assert_close(g,torch.ones(1,2))
 def test_straight_through_and_normalization(self):
  torch.manual_seed(1);m=SharedGroups(4,6,3,2,3);x=torch.randn(7,4);r=m(x)
  self.assertTrue(torch.all((r['support']==0)|(r['support']==1)))
  loss=m.objective(x,r,.01,.01,.01);loss.backward()
  for p in [m.dictionary,m.coordinate_encoder.weight,m.group_encoder.weight,m.membership_logits]:
   self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0.)
  torch.testing.assert_close(m.dictionary_matrix().norm(dim=0),torch.ones(6))
 def test_oracle_shared_dictionary_unseen_combinations(self):
  torch.manual_seed(99);B=torch.linalg.qr(torch.randn(12,5,dtype=torch.float64)).Q
  M=torch.tensor([[1,1,1,0,0],[0,0,1,1,1]],dtype=torch.float64)
  m=SharedGroups(12,5,2,2,3).double()
  with torch.no_grad():
   m.dictionary.copy_(B);m.coordinate_encoder.weight.copy_(B.T);m.coordinate_encoder.bias.zero_();m.bias.zero_()
  a=torch.randn(100,5,dtype=torch.float64);g=torch.ones(100,2,dtype=torch.float64);x=a@B.T
  out=m(x,gates=g,membership=M)['reconstruction'];torch.testing.assert_close(out,x,atol=1e-12,rtol=1e-12)
  # Unseen joint activity still decodes the shared component once.
  removed=m(x,gates=g,membership=M,remove_atoms=(2,))['reconstruction']
  torch.testing.assert_close(x-removed,a[:,2,None]*B[:,2],atol=1e-12,rtol=1e-12)
 def test_fixed_support_coordinate_gradient(self):
  torch.manual_seed(100);m=SharedGroups(4,5,2,2,3).double();x=torch.randn(2,4,dtype=torch.float64);a=torch.randn(2,5,dtype=torch.float64,requires_grad=True)
  M=torch.tensor([[1,1,1,0,0],[0,0,1,1,1]],dtype=torch.float64);g=torch.ones(2,2,dtype=torch.float64)
  self.assertTrue(torch.autograd.gradcheck(lambda coords:m(x,gates=g,coordinates=coords,membership=M)['reconstruction'],(a,)))
 def test_closure_is_idempotent(self):
  graph=[(),(),(0,1),(2,)];g=torch.tensor([[0.,0.,0.,1.],[1.,0.,0.,0.]])
  closed=implication_closure(g,graph);torch.testing.assert_close(implication_closure(closed,graph),closed)
 def test_shared_computation_and_selective_edges(self):
  # u=x0*x1 is evaluated once; outputs 2u,3u reuse it, x0 independent.
  m=ComputationDAG(2,3,[(),(),(0,1),(2,),(2,)],products=[(),(),((0,1),),(),()])
  with torch.no_grad():
   for f in m.functions:f.weight.zero_();f.bias.zero_()
   m.functions[0].weight[0,0]=1;m.functions[1].weight[0,1]=1
   m.functions[2].weight[0,-1]=1;m.functions[3].weight[0,-1]=2;m.functions[4].weight[0,-1]=3
   m.readout.weight.zero_();m.readout.weight[0,3]=1;m.readout.weight[1,4]=1;m.readout.weight[2,0]=1
  calls=[0]*5;hooks=[]
  for i,f in enumerate(m.functions):
   def count(mod,args,out,i=i):calls[i]+=1
   hooks.append(f.register_forward_hook(count))
  x=torch.tensor([[2.,4.],[-1.,3.]]);g=torch.ones(2,5);y=m(x,gates=g)['reconstruction']
  for h in hooks:h.remove()
  self.assertEqual(calls,[1]*5);torch.testing.assert_close(y,torch.tensor([[16.,24.,2.],[-6.,-9.,-1.]]))
  no=m(x,gates=g,remove_nodes=(2,))['reconstruction'];torch.testing.assert_close(no[:,:2],torch.zeros(2,2));torch.testing.assert_close(no[:,2],x[:,0])
  edge=m(x,gates=g,remove_edges=((2,3),))['reconstruction'];torch.testing.assert_close(edge[:,0],torch.zeros(2));torch.testing.assert_close(edge[:,1:],y[:,1:])

if __name__=='__main__':unittest.main()
