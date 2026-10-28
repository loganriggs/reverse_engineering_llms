from pathlib import Path


def test_multimetric_quantization_includes_all_mechanistic_levels():
    source = Path("scripts/run_bilin18_layer8_multimetric_quantization.py").read_text()
    for required in ("direct_mlp", "unweighted_logit", "softmax_fisher", "cross_metric_overlap", "oracle_capture", "routed_capture"):
        assert required in source
