import sys
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from run_bilin18_destination_fisher_geometry import cosine_gram


def test_cosine_gram_normalizes_diagonal():
    gram=np.asarray([[4.0,1.0],[1.0,9.0]])
    cosine=cosine_gram(gram)
    assert np.allclose(np.diag(cosine),1)
    assert abs(cosine[0,1]-1/6)<1e-12
