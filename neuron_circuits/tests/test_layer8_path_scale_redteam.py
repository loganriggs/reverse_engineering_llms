import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_path_scale_redteam import at  # noqa: E402


def test_at_reads_nested_path_metric():
    result = {
        "0.1": {
            "fineweb": {
                "summary": {
                    "within_route": {
                        "oracle": {"additive": {"metric": 0.75}},
                    },
                },
            },
        },
    }
    assert at(result, 0.1, "fineweb", "oracle", "additive", "metric") == 0.75
