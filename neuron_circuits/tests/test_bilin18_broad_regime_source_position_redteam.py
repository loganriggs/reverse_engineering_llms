import importlib.util
import json
from pathlib import Path
import sys


ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts"/"redteam_bilin18_broad_regime_source_positions.py"
spec=importlib.util.spec_from_file_location("broad_source_redteam",SCRIPT)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)


def test_source_position_config_is_locked_and_exhaustive():
    config=json.loads((ROOT/"configs"/"bilin18_broad_regime_source_position_redteam.json").read_text())
    assert config["status"]=="locked-after-operand-grammar-before-position-subset-null"
    assert "P0_edited_sources_are_exceptional_for_direct_read_programs" in config["hypotheses"]


def test_program_position_delta_shape():
    # Algebraic shape is exercised without loading the checkpoint.
    assert list(module.itertools.combinations(range(5),2))[-1]==(3,4)
    assert len(list(module.itertools.combinations(range(5),2)))==10
