import json
from pathlib import Path


def test_conditioned_spike_experiment_is_locked_and_rotation_invariant():
    config = json.loads(Path(
        "configs/bilin18_layer3_conditioned_spectral_spikes.json"
    ).read_text())
    source = Path(
        "scripts/run_bilin18_layer3_conditioned_spectral_spikes.py"
    ).read_text()
    assert "locked" in config["status"]
    assert config["output_rank"] == 4
    assert config["null_replicates"] >= 8
    assert config["random_output_directions"] >= 32
    for required in (
        "symmetric_channel_gram", "output_basis", "canonical", "projector_error",
        "fit_powerlaw", "sign_null", "random_output", "top4_energy_fraction",
    ):
        assert required in source
