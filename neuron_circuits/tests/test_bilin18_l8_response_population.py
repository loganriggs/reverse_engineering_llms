import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_bilin18_l8_response_population as experiment

def test_response_mediation_holds_upstream_removal_fixed(monkeypatch):
    torch.manual_seed(31)
    dim=18;heads=9
    identity=torch.nn.Linear(dim,dim,bias=False)
    with torch.no_grad():identity.weight.copy_(torch.eye(dim))
    block=SimpleNamespace(lambdas=[1.,0.],attn=SimpleNamespace(c_proj=identity),
        mlp=SimpleNamespace(Left=identity,Right=identity,Down=identity,Down_bias=torch.zeros(dim)))
    model=SimpleNamespace(config=SimpleNamespace(n_embd=dim,n_head=heads),embedding=torch.nn.Embedding(5,dim),blocks=[block],lm_head=identity,tt=None)
    def details(attn,x,bus,config,tt):
        b,t,_=x.shape
        return {'write':x,'mixed_value':x.reshape(b,t,heads,2),
                'pattern':torch.eye(t).expand(b,heads,t,t),'value_bus':torch.zeros(b,t,dim)}
    monkeypatch.setattr(experiment,'attention_details',details)
    ids=torch.tensor([[0,1],[2,3]])
    result,delta=experiment.finite_panel(model,ids,3,{'identity':torch.eye(dim),'zero':torch.zeros(0,dim)},
        {'layer':0,'finite_condition_chunk':2})
    torch.testing.assert_close(result['identity'],result['gold'],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(result['zero'],result['no_response'])
    # Independent one-block forward: delete only head3's two destination channels.
    x=F.rms_norm(model.embedding(ids),(dim,));attention=F.rms_norm(x,(dim,));changed_attention=attention.clone();changed_attention[:,-1,6:8]=0
    pre=x+attention;changed_pre=x+changed_attention
    native_write=F.rms_norm(pre,(dim,)).square();changed_write=F.rms_norm(changed_pre,(dim,)).square()
    def logits(state):return 30*torch.tanh(F.rms_norm(state,(dim,))[:,-1]/30)
    torch.testing.assert_close(result['gold'],logits(changed_pre+changed_write))
    torch.testing.assert_close(result['no_response'],logits(changed_pre+native_write))
    torch.testing.assert_close(delta,changed_write[:,-1]-native_write[:,-1])
