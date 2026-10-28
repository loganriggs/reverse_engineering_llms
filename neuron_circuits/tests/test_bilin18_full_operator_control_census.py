import sys
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from run_bilin18_full_operator_control_census import complete_controls


def test_complete_controls_use_every_nonpair_operator():
    gram=np.eye(5);gram[0,1]=gram[1,0]=.2
    scale,controls,cosine=complete_controls(gram,np.ones(5),0,1)
    assert scale==1 and cosine==.2
    assert {row["other"] for row in controls}=={2,3,4}
