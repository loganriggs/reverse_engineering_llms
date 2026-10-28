import torch
from bilin18_readout_feature_families import fit_features,features

def test_features_use_training_only_and_obey_read_span():
    torch.manual_seed(12);x=torch.randn(320,160);q=torch.linalg.qr(torch.randn(160,64)).Q;p={'mean':torch.zeros(160),'core':q[:,:32],'partner':q[:,32:]}
    for family in ['quadratic','outside']:
        f,m=fit_features(x,p,family,256);changed=x.clone();changed[256:]+=20
        _,m2=fit_features(changed,p,family,256)
        for k in m:assert torch.equal(m[k],m2[k])
        assert torch.allclose(features(x,p,m),f)
        delta=torch.randn_like(x);inside=(delta@q)@q.T;outside=delta-inside
        perturb=outside if family=='quadratic' else inside
        assert torch.allclose(features(x+perturb,p,m),f,atol=3e-4,rtol=3e-4)
