import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))

from run_bilin12_attention_destination_metric_redteam import GROUPS


def test_destination_groups_are_three_disjoint_roles():
    assert GROUPS==("edited_position","subsequent_context","final_position")
