import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l8_channel_coordination import permuted_weights,canonical_weights
from run_bilin18_l8_context_role_compression import B

def test_joint_permutation_preserves_function_and_separate_shuffles_preserve_grams():
    torch.manual_seed(4)
    L=torch.randn(11,7,dtype=torch.double);R=torch.randn_like(L);D=torch.randn(5,11,dtype=torch.double);p=torch.randperm(11);x=torch.randn(13,7,dtype=torch.double)
    joint=permuted_weights(L,R,D,p,'joint')
    torch.testing.assert_close(B(x,x,*joint),B(x,x,L,R,D))
    _,Rp,_=permuted_weights(L,R,D,p,'right');_,_,Dp=permuted_weights(L,R,D,p,'down')
    torch.testing.assert_close(Rp.T@Rp,R.T@R);torch.testing.assert_close(Dp@Dp.T,D@D.T)
    assert (B(x,x,L,Rp,D)-B(x,x,L,R,D)).norm()>1

def test_canonicalization_is_invariant_to_signed_scaling_and_pair_exchange():
    torch.manual_seed(9)
    L=torch.randn(11,7,dtype=torch.double);R=torch.randn_like(L);D=torch.randn(5,11,dtype=torch.double)
    a=torch.exp(torch.randn(11,dtype=torch.double))*torch.sign(torch.randn(11));b=torch.exp(torch.randn(11,dtype=torch.double))*torch.sign(torch.randn(11))
    L2=L*a[:,None];R2=R*b[:,None];D2=D/(a*b)
    swap=torch.arange(11)%2==0;L3=torch.where(swap[:,None],R2,L2);R3=torch.where(swap[:,None],L2,R2)
    canonical=canonical_weights(L,R,D);transformed=canonical_weights(L3,R3,D2)
    for x,y in zip(canonical,transformed):torch.testing.assert_close(x,y)
    x=torch.randn(13,7,dtype=torch.double);torch.testing.assert_close(B(x,x,*canonical),B(x,x,L,R,D))
    p=torch.randperm(11);_,_,changed=permuted_weights(*canonical,p,'direction')
    torch.testing.assert_close(changed.norm(dim=0),canonical[2].norm(dim=0))
