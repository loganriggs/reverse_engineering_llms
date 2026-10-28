import json
from pathlib import Path


def test_source_router_result_exists():
    config = json.loads(Path("configs/bilin12_layer01_all_chart_router_analysis.json").read_text())
    assert Path(config["source_results"]).exists()
