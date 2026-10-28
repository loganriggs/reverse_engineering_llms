import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_broad_regime_construction.py"
spec = importlib.util.spec_from_file_location("broad_regime_construction", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_shapley_recomposes_arbitrary_three_factor_game():
    scores = {}
    for world in module.WORLDS:
        scores[world] = module.torch.tensor([world[0] + 2 * world[1] - world[2] + 3 * world[0] * world[1]], dtype=module.torch.float64)
    phi = module.shapley(scores)
    module.torch.testing.assert_close(sum(phi.values()), scores[(1, 1, 1)] - scores[(0, 0, 0)])


def test_repeated_sequence_pairs_change_payload_not_key():
    pairs = module.repeated_sequence_pairs()
    assert len(pairs) == len(module.COLORS)
    for base, source in pairs:
        assert base.answer != source.answer
        assert base.prompt.split()[0] == source.prompt.split()[0]
        assert base.candidates == source.candidates


def test_locked_suite_uses_broad_operational_families():
    config = json.loads((ROOT / "configs" / "bilin18_broad_regime_construction.json").read_text())
    assert config["status"] == "locked-after-broad-competence-redteam-before-construction-results"
    assert len(config["families"]) == 7
    assert "ioi" in config["families"]
    assert "capital_recall" in config["families"]
    assert "repeated_sequence" in config["families"]
    assert config["layers"] == list(range(18))


def test_length_batches_never_mix_sequence_lengths():
    pairs = [
        {"base_ids": [1, 2], "source_ids": [3, 4]},
        {"base_ids": [1, 2, 3], "source_ids": [4, 5, 6]},
        {"base_ids": [7, 8], "source_ids": [9, 10]},
    ]
    batches = list(module.length_batches(pairs, 4))
    assert sorted(len(batch) for batch in batches) == [1, 2]
    assert all(len({len(pair["base_ids"]) for pair in batch}) == 1 for batch in batches)
