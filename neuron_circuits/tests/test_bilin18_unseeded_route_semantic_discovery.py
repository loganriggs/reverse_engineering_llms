import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_unseeded_route_semantic_discovery.py"
spec = importlib.util.spec_from_file_location("unseeded_route_semantic_discovery", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_select_candidates_applies_margin_and_layer_gate():
    base = {"layer": 1, "cluster": 0, "center_cosine": .95, "discovery_size": 100, "replication_size": 100, "discovery_member_cosine": .8, "replication_member_cosine": .8, "discovery_margin": .2, "replication_margin": .2}
    geometry = {"clusters": {"1": [base, {**base, "cluster": 1, "discovery_margin": .01}]}}
    redteam = {"hypotheses": {"R1_k4_centers_are_well_separated": {"layers": [1]}}}
    config = {"minimum_center_cosine": .9, "minimum_cluster_size": 64, "minimum_member_cosine": .7, "minimum_margin": .1, "maximum_candidates": 16}
    result = module.select_candidates(geometry, redteam, config)
    assert len(result) == 1 and result[0]["cluster"] == 0
