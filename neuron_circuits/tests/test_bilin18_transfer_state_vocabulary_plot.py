from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_transfer_state_vocabulary_plot_exists():
    assert (ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_transfer_state_vocabulary.png").exists()
