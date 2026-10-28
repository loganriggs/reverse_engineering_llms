import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_probability_support import summarize

def test_stratum_recovery_pools_error_and_bootstraps_documents():
    base=np.array([1.,9.,2.,18.]);kl=np.array([0.,9.,1.,18.]);mask=np.array([True,False,True,False])
    ix=np.array([[0,1],[0,0],[1,1]])
    result=summarize(kl,base,mask,ix)
    assert result['sites']==2
    np.testing.assert_allclose(result['recovery'],2/3)
    np.testing.assert_allclose(result['document_bootstrap95'],np.quantile([2/3,1.,.5],[.025,.975]))
    assert summarize(kl,base,np.zeros(4,dtype=bool),ix)=={'sites':0}
