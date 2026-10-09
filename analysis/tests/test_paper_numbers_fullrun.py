"""Paper number macros: formatting is exact, negative values are set in math mode, macro names carry no digits."""
import math
import re
from pathlib import Path

import pytest

from analysis.scripts import paper_numbers_fullrun as pn

BUNDLE = Path.home() / "Hamburg/geodml-inputs/fullrun-run2"


def test_formatters_round_and_mark_negatives():
    assert pn.slope(0.0717) == "+0.072" and pn.slope(-0.0003, 4) == "\\ensuremath{-0.0003}"
    assert pn.ci(0.067, 0.076) == "[+0.067, +0.076]" and pn.ci(-0.0014, 0.0008, 4) == "\\ensuremath{[-0.0014, +0.0008]}"
    assert pn.pct(0.8897) == "89\\%" and pn.pct(-0.044) == "\\ensuremath{-4}\\%"
    assert pn.pct1(-0.0004) == "0.0\\%" and pn.pct1(0.024) == "2.4\\%"
    assert pn.pctci(0.86, 0.92) == "[86\\%, 92\\%]"
    feature = {"odds_ratio_per_sd": math.exp(0.1), "ci95": [0.05, 0.15]}
    assert pn.orci(feature) == "1.11 [1.05, 1.16]"
    assert pn.num(27965) == "27{,}965" and pn.pval(None) == "--" and pn.pval(0.004975) == "0.005"


def test_block_names_carry_no_digits():
    for blk, _ in pn.BLOCKS:
        assert re.fullmatch(r"[A-Za-z]+", pn.TEX_BLOCK[blk])


@pytest.mark.skipif(not (BUNDLE / "verdicts.json").exists(), reason="full-run bundle not on this machine")
def test_generated_macros_are_valid_control_sequences(tmp_path):
    pn.main([str(BUNDLE), str(tmp_path)])
    text = (tmp_path / "numbers-heldout.tex").read_text()
    names = re.findall(r"\\newcommand\{\\([^}]+)\}", text)
    assert len(names) == len(set(names)) > 1000
    assert all(re.fullmatch(r"[A-Za-z]+", n) for n in names)
    tables = (tmp_path / "A3-generated-tables.tex").read_text()
    # verdicts, predictions, decomposition, the compact odds-ratio table and the block-loss table
    assert tables.count("\\begin{table*}") == tables.count("\\end{table*}") == 5
    for name in ("inclusion_models_full.csv", "inclusion_models_blocks.csv", "keep_order_models_full.csv"):
        rows = (tmp_path / "supplement" / name).read_text().splitlines()
        assert len(rows) > 100 and rows[0].count(",") >= 7
