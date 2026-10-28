import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_bilin18_layer17_tail_routed_local_field import selected_subspace_cosine  # noqa: E402


def test_selected_subspace_cosine_is_one_for_same_selected_spaces():
    generator = torch.Generator().manual_seed(15)
    q = torch.linalg.qr(torch.randn(3, 12, 4, generator=generator)).Q
    labels = torch.tensor([0, 2, 1, 0, 1])
    values = selected_subspace_cosine(q, labels, q, labels)
    assert torch.all(values > 0.999)


def test_local_field_source_has_label_free_and_null_controls():
    source = Path("scripts/run_bilin18_layer17_tail_routed_local_field.py").read_text()
    for required in ("selected_subspace_cosine", "shuffled_context", "route_optimal", "exact_output_kl", "disagreement"):
        assert required in source
