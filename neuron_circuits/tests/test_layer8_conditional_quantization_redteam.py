from pathlib import Path


def test_high_resolution_redteam_has_required_controls():
    source = Path("scripts/run_bilin18_layer8_conditional_quantization_redteam.py").read_text()
    for required in ("centroid", "medoid", "random", "nearest_context", "training_similarity", "oracle_capture", "routed_capture"):
        assert required in source
