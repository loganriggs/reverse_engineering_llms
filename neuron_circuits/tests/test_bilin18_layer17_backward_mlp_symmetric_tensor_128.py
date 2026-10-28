import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer17_backward_mlp_symmetric_tensor_128.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer17_backward_mlp_symmetric_tensor_128.json"
TENSOR = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer17_backward_mlp_symmetric_tensor_128.pt"


def test_rank128_tensor_scaling_is_locked():
    config = json.loads(CONFIG.read_text())
    assert config["input_dimension"] == config["output_dimension"] == 128
    assert config["mode_budgets"][-1] == 128
    assert Path(ROOT / config["rank64_result_path"]).exists()


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
    assert TENSOR.exists()
