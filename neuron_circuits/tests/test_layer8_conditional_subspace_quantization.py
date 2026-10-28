import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_bilin18_layer8_conditional_subspace_quantization import grassmann_similarity, prototype_direction  # noqa: E402


def test_grassmann_similarity_and_prototype_direction():
    generator = torch.Generator().manual_seed(7)
    q = torch.linalg.qr(torch.randn(3, 10, 2, generator=generator)).Q
    similarity = grassmann_similarity(q, q)
    assert similarity.shape == (3, 3)
    assert torch.all(torch.diag(similarity) > 0.999)
    response = torch.randn(20, 10, generator=generator)
    direction = prototype_direction(response, q[0])
    assert direction.shape == (10,)
    assert abs(float(direction.norm()) - 1) < 1e-5


def test_middle_quantization_uses_oracle_router_and_finite_controls():
    source = Path("scripts/run_bilin18_layer8_conditional_subspace_quantization.py").read_text()
    for required in ("grassmann_kmeans", "oracle_capture", "routed_capture", "random_capture", "finite_fidelity", "route_match"):
        assert required in source
