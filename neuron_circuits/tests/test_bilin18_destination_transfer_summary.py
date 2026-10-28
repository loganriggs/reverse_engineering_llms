from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_destination_transfer_summary_inputs_and_figure_exist():
    artifact = ROOT / "artifacts/bilin18_tensor_decomposition"
    for name in (
        "bilin18_destination_path_predictor.json",
        "bilin18_destination_attention_order_exact.json",
        "bilin18_destination_transfer_conditioning.json",
        "bilin18_destination_transfer_summary.png",
    ):
        assert (artifact / name).exists()
