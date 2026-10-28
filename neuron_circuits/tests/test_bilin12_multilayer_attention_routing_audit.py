import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_multilayer_attention_routing_audit import routes


def test_routes_preserve_ranked_prefixes_and_cover_all_layers():
    ranking=[6,5,4,8,2,10,1,3,9,7,11,0]; result=routes(ranking)
    assert result["top1"] == [6]
    assert result["top4"] == [6,5,4,8]
    assert set(result["all_ranked"]) == set(range(12))
    assert result["all_layers"] == list(range(12))
