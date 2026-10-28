import sys
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from plot_bilin18_function_ontology_summary import cumulative


def test_cumulative_normalizes_spectrum():
    values=cumulative([3,1])
    assert np.allclose(values,[.75,1])
