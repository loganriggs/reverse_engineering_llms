import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_layers0_3_compositional_ontology import (  # noqa: E402
    positive_mass_statistics, specifications,
)


def test_layers0_3_replication_is_locked_and_uses_all_pairs():
    config = json.loads(
        (ROOT / "configs/bilin18_layers0_3_compositional_ontology_replication.json").read_text()
    )
    assert config["status"] == "locked-replication-after-16-site-pilot"
    assert config["source_layers"] == [0, 1, 2, 3]
    assert config["sites_per_feature"] == 12
    assert config["coalition_ranks"] == [1, 4, 16, 64, 256, 1024]


def test_source_layer_changes_suffix_without_changing_role_ontology():
    layer0 = specifications(0, 18, [4, 8])
    layer3 = specifications(3, 18, [4, 8])
    assert layer0["all_full"]["layers"] == set(range(1, 18))
    assert layer3["all_full"]["layers"] == set(range(4, 18))
    assert layer3["near_4"]["layers"] == {4, 5, 6, 7}
    assert layer3["query_key"]["roles"] == {"q1", "k1", "q2", "k2"}


def test_positive_mass_k90_ignores_harmful_cancellation():
    result = positive_mass_statistics(torch.tensor([4.0, 3.0, 2.0, 1.0, -100.0]))
    assert result["positive_count"] == 4
    assert result["positive_k90"] == 3
    assert result["positive_mass"] == 10.0
