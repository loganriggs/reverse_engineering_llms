import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts/run_bilin18_attention_layer_path_program.py"
CONFIG = ROOT / "configs/bilin18_attention_layer_path_program.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_attention_layer_path_program.json"

spec = importlib.util.spec_from_file_location("attention_paths", SCRIPT)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_attention_path_inventory_is_complete():
    assert len(module.ATTENTION_PATHS) == 32
    assert len(set(module.ATTENTION_PATHS)) == 32
    assert module.ATTENTION_PATHS[0] == ("I",) * 5


def test_sequence_rms_jvp_matches_autograd():
    generator = torch.Generator().manual_seed(34)
    x = torch.randn(2, 5, 8, generator=generator)
    tangents = torch.randn(2, 3, 5, 8, generator=generator)
    expected = []
    for index in range(tangents.shape[1]):
        _, tangent = torch.func.jvp(lambda value: torch.nn.functional.rms_norm(value, (8,)),
                                    (x,), (tangents[:, index],))
        expected.append(tangent)
    torch.testing.assert_close(module.sequence_rms_jvp(x, tangents), torch.stack(expected, 1), rtol=1e-5, atol=1e-6)


def test_output_matches_locked_config_if_present():
    if not OUTPUT.exists(): return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert len(payload["path_order"]) == 32
