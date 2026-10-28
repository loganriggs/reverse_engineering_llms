import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_broad_regime_operand_grammar.py"
spec = importlib.util.spec_from_file_location("broad_operand_grammar", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [7]


def test_role_masks_partition_positions_once():
    probe = type("P", (), {"candidates": (" a", " b"), "answer": " a", "prompt": "x"})()
    pair = {"base_ids": [1, 2, 3, 4], "source_ids": [1, 5, 3, 4], "base": probe, "source": probe}
    original = module.class_token_ids
    module.class_token_ids = lambda tokenizer, item: [[2], [5]]
    try:
        roles = ["edited_non_destination", "answer_mentions", "query_repeats", "other", "destination"]
        masks = module.role_masks(pair, FakeTokenizer(), roles, "cpu")
        stacked = module.torch.stack([masks[name] for name in roles])
        assert module.torch.equal(stacked.sum(0), module.torch.ones(4, dtype=module.torch.long))
        assert masks["edited_non_destination"].tolist() == [False, True, False, False]
        assert masks["destination"].tolist() == [False, False, False, True]
    finally:
        module.class_token_ids = original


def test_five_factor_shapley_recomposes():
    names = tuple("abcde")
    scores = {world: module.torch.tensor([sum((index + 1) * bit for index, bit in enumerate(world))], dtype=module.torch.float64) for world in module.itertools.product((0, 1), repeat=5)}
    phi = module.shapley(scores, names)
    module.torch.testing.assert_close(sum(phi.values()), scores[(1, 1, 1, 1, 1)] - scores[(0, 0, 0, 0, 0)])


def test_operand_config_is_locked_and_roles_are_disjoint_schema():
    config=json.loads((ROOT/"configs"/"bilin18_broad_regime_operand_grammar.json").read_text())
    assert config["status"] == "locked-after-broad-subset-null-before-operand-results"
    assert config["roles"] == ["edited_non_destination", "answer_mentions", "query_repeats", "other", "destination"]
