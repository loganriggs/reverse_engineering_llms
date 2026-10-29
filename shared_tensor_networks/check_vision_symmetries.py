"""Check exact restrictions shared by normalized and polynomial vision models."""
import json
from pathlib import Path
import torch
from vision_circuits import BilinearCircuit

def main():
    torch.set_num_threads(1)
    x=torch.randn(11,5,generator=torch.Generator().manual_seed(99),dtype=torch.float64)
    rows=[]
    for depth in [1,2,3,4]:
        for tree in [False,True]:
            for norm in [False,True]:
                model=BilinearCircuit(5,4,depth,3,tree,norm,10,torch.float64)
                # Break equality of occurrence values so this checks genuinely untied trees.
                if tree:
                    with torch.no_grad():
                        for i,a in enumerate(model.A):a.add_(0.003*i)
                y=model(x)
                torch.testing.assert_close(model(-x),y,rtol=0,atol=0)
                row=dict(depth=depth,tree=tree,rmsnorm=norm,evenness=True)
                if not norm:
                    scaled=model(1.25*x);expected=1.25**(2**depth)*y
                    torch.testing.assert_close(scaled,expected,rtol=1e-10,atol=1e-10)
                    assert torch.equal(scaled.argmax(1),y.argmax(1))
                    row['input_degree']=2**depth
                rows.append(row)
    # Exact single-node RMSNorm response to positive input rescaling.
    z=torch.randn(11,4,generator=torch.Generator().manual_seed(13),dtype=torch.float64)
    c=1.7;eps=1e-6
    direct=c*c*z/torch.sqrt((c*c*z).square().mean(-1,keepdim=True)+eps)
    equivalent=z/torch.sqrt(z.square().mean(-1,keepdim=True)+eps/c**4)
    torch.testing.assert_close(direct,equivalent,rtol=1e-13,atol=1e-13)
    out=Path(__file__).parent/'results/vision_correctness/symmetries.json'
    out.write_text(json.dumps(dict(status='passed',rows=rows,rmsnorm_rescaling_identity=True),indent=2)+'\n')
    print('Passed 16 architecture/depth/normalization cases and RMSNorm scaling identity')
if __name__=='__main__':main()
