"""The full run's first-look checker reproduces the verdicts of the stored Mac parts and flags missing pieces."""

import json
from pathlib import Path
import shutil

import pytest

from analysis.steelman import check_fullrun

MAC = Path.home() / "Hamburg/geodml-inputs/steelman-v1"


@pytest.mark.skipif(not (MAC / "chain.json").exists(), reason="Mac steelman outputs not present")
def test_checker_on_mac_exploration_parts(tmp_path):
    folder = tmp_path / "steelman-exploration"
    folder.mkdir()
    for f in MAC.glob("*.json"):
        shutil.copy(f, folder / f.name)
    rc = check_fullrun.main([str(tmp_path), "--out", str(tmp_path / "v.json")])
    r = json.loads((tmp_path / "v.json").read_text())
    assert rc == 1                                                   # confirmation split and new parts are missing
    assert {v["verdict"] for v in r["exploration"]["C1"].values()} == {"supported"}
    assert r["exploration"]["C2"]["qwen38"]["verdict"] == "supported"
    assert r["exploration"]["beta_K"]["qwen38 · Reactive"]["estimate"] > 0
    assert "supply" in r["exploration"]["missing_parts"] and r["confirmation"] == {"missing": True}
