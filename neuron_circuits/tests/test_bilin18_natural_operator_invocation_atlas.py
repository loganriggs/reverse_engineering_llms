import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_natural_operator_invocation_atlas.py"
spec = importlib.util.spec_from_file_location("natural_operator_invocation_atlas", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_cosine_rows():
    routes = torch.tensor([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]])
    result = module.cosine_rows(routes, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(result, torch.tensor([1.0, 0.0, -1.0]))


def test_heap_retains_largest_scores():
    heap = []
    for index, score in enumerate([1.0, 4.0, 2.0, 3.0]):
        module.push(heap, score, {"global_row": index, "destination": 1, "source": 0}, 2)
    assert sorted(item[0] for item in heap) == [3.0, 4.0]
