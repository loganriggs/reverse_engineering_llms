"""Expand the downstream MLP in the post-fit sparse A/B coordinates."""

from __future__ import annotations

import json

import numpy as np
import torch

from dual_joint_function_canonical import HERE
from train_xor_count_language import XorCountAttention,examples


def coefficients(model,row,indices,q):
    transform=np.eye(3)
    transform[:2,:2]=row[2*q:2*q+2]
    left=model.mlp_left.weight.detach().numpy()@transform
    right=model.mlp_right.weight.detach().numpy()@transform
    read=model.read.weight.detach().numpy()
    # Expand y = sum_j read_j (left_j dot [z0,z1,1])
    #                          (right_j dot [z0,z1,1]) + bias.
    a,b=indices
    cross=np.sum(read*(left[:,a]*right[:,b]+left[:,b]*right[:,a])[None,:],axis=1)
    square_a=np.sum(read*(left[:,a]*right[:,a])[None,:],axis=1)
    square_b=np.sum(read*(left[:,b]*right[:,b])[None,:],axis=1)
    linear_a=np.sum(read*(left[:,a]*right[:,2]+left[:,2]*right[:,a])[None,:],axis=1)
    linear_b=np.sum(read*(left[:,b]*right[:,2]+left[:,2]*right[:,b])[None,:],axis=1)
    constant=np.sum(read*(left[:,2]*right[:,2])[None,:],axis=1)+model.read.bias.detach().numpy()
    return dict(cross=cross.tolist(),square_A=square_a.tolist(),
                square_B=square_b.tolist(),linear_A=linear_a.tolist(),
                linear_B=linear_b.tolist(),constant=constant.tolist())


def run():
    sparse=json.loads((HERE/"xor_count_sparse.json").read_text())
    rows=[]
    for record in sparse["rows"]:
        seed=record["seed"]
        model=XorCountAttention().double()
        model.load_state_dict(torch.load(HERE/f"xor_count_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        model.eval()
        row=np.array(record["row_factors"])
        dictionary=np.array(record["dictionary"])
        indices=(record["match"]["A"],record["match"]["B"])
        queries=[coefficients(model,row,indices,q) for q in (0,1)]
        # Check the symbolic expansion against the actual downstream module.
        probe=np.array([[-2.1,.4],[.3,1.7],[2.,-1.3]])
        for q in (0,1):
            c=queries[q]; a=probe[:,indices[0]];b=probe[:,indices[1]]
            expanded=(np.outer(a*b,c["cross"])+
                np.outer(a*a,c["square_A"])+np.outer(b*b,c["square_B"])+
                np.outer(a,c["linear_A"])+np.outer(b,c["linear_B"])+
                np.array(c["constant"]))
            heads=probe@row[2*q:2*q+2].T
            with torch.no_grad():
                h=torch.from_numpy(heads)
                hidden=torch.cat((h,torch.ones_like(h[:,:1])),dim=1)
                actual=model.read(model.mlp_left(hidden)*model.mlp_right(hidden)).numpy()
            assert np.max(np.abs(expanded-actual))<1e-9
        # Independent, varied-count test: does the cross monomial alone
        # (with its learned sign) predict XOR? This is diagnostic, not a refit.
        tokens,targets,attrs=examples(30000,8000+seed,False)
        tokens=tokens.numpy();targets=targets.numpy().astype(bool)
        counts=np.stack([(tokens[:,:-1]==key).sum(1) for key in range(6)],axis=1)
        z=counts@dictionary.T
        q=tokens[:,-1]-6
        pure=np.array([queries[qi]["cross"][2] for qi in q])*z[:,indices[0]]*z[:,indices[1]]
        xor_accuracy=float(np.mean((pure>0)==targets[:,2]))
        rare=float(np.mean((pure[attrs["heldout"]]>0)==targets[attrs["heldout"],2]))
        dominance=[abs(c["cross"][2])/(abs(c["square_A"][2])+
                   abs(c["square_B"][2])+1e-12) for c in queries]
        rows.append(dict(seed=seed,queries=queries,
            xor_cross_over_square_sum=dominance,
            cross_only_xor_accuracy=xor_accuracy,
            cross_only_rare_xor_accuracy=rare))
        print(seed,dominance,xor_accuracy,rare,flush=True)
    output=dict(n_seeds=10,
        fit="no fit: algebraic expansion of original bilinear MLP after exact rank-two weight-only cut and label-free sparse gauge",
        evaluation="A/B matching and cross-only XOR accuracy use labels after fitting",
        rows=rows)
    (HERE/"xor_count_polynomial.json").write_text(json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":run()
