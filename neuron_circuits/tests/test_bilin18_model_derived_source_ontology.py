import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_model_derived_source_ontology import spherical_kmeans


def test_weighted_spherical_kmeans_separates_obvious_directions():
    features = torch.tensor([[1.0, 0.0], [.9, .1], [-1.0, 0.0], [-.9, -.1]])
    assignments, _, distortion = spherical_kmeans(features, torch.ones(4), 2, 10, 3, 7)
    assert assignments[0] == assignments[1]
    assert assignments[2] == assignments[3]
    assert assignments[0] != assignments[2]
    assert distortion < .01
