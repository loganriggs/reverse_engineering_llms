import json
from pathlib import Path


def test_lexical_regime_hierarchy_is_posthoc_and_hierarchical():
    config = json.loads(Path(
        "configs/bilin18_layer3_lexical_regime_hierarchy.json"
    ).read_text())
    source = Path(
        "scripts/analyze_bilin18_layer3_lexical_regime_hierarchy.py"
    ).read_text()
    assert config["status"].startswith("posthoc-")
    assert config["bootstrap_replicates"] >= 1000
    for required in (
        "occurrence_weighted_median", "pair_balanced_median", "pair_bootstrap_95",
        "median_within_pair_sign_consistency", "cross_corpus", "aggregation_sign_agrees",
    ):
        assert required in source
