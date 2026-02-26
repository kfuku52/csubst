import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "evaluate_epistasis_simulation.py"
_spec = importlib.util.spec_from_file_location("evaluate_epistasis_simulation", _TOOL_PATH)
_tool = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_tool)


def test_calc_auc_perfect_separation():
    auc = _tool._calc_auc(scores=np.array([0.1, 0.2, 0.8, 0.9]), labels=np.array([0, 0, 1, 1], dtype=bool))
    assert auc == pytest.approx(1.0, abs=1e-12)


def test_calc_auc_tie_returns_half():
    auc = _tool._calc_auc(scores=np.array([1.0, 1.0, 1.0, 1.0]), labels=np.array([0, 1, 0, 1], dtype=bool))
    assert auc == pytest.approx(0.5, abs=1e-12)


def test_average_precision_and_precision_at_k():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1], dtype=bool)
    ap = _tool._calc_average_precision(scores=scores, labels=labels)
    p_at_2 = _tool._calc_precision_at_k(scores=scores, labels=labels, k=2)
    assert ap == pytest.approx(1.0, abs=1e-12)
    assert p_at_2 == pytest.approx(1.0, abs=1e-12)


def test_calc_fg_detection_metrics():
    df = pd.DataFrame(
        {
            "branch_id_1": [0, 0, 1, 1],
            "branch_id_2": [2, 3, 2, 3],
            "is_fg": ["N", "N", "Y", "Y"],
            "omegaCany2spe": [0.2, 0.1, 2.0, 3.0],
        }
    )
    out = _tool._calc_fg_detection_metrics(df=df, score_col="omegaCany2spe")
    assert out["n_total_valid"] == 4
    assert out["n_pos"] == 2
    assert out["n_neg"] == 2
    assert out["auroc"] == pytest.approx(1.0, abs=1e-12)
    assert out["average_precision"] == pytest.approx(1.0, abs=1e-12)


def test_calc_similarity_to_baseline_jaccard_and_pearson():
    base = pd.DataFrame(
        {
            "branch_id_1": [0, 1, 2],
            "branch_id_2": [3, 4, 5],
            "omegaCany2spe": [1.0, 10.0, 5.0],
        }
    )
    cur = pd.DataFrame(
        {
            "branch_id_1": [0, 1, 2],
            "branch_id_2": [3, 4, 5],
            "omegaCany2spe": [1.0, 8.0, 6.0],
        }
    )
    out = _tool._calc_similarity_to_baseline(
        df_base=base,
        df_cur=cur,
        score_col="omegaCany2spe",
        jaccard_threshold=5.0,
    )
    assert out["n_pair"] == 3
    assert out["intersection"] == 2
    assert out["union"] == 2
    assert out["jaccard"] == pytest.approx(1.0, abs=1e-12)
    assert np.isfinite(out["pearson_r_log10"])


def test_parse_time_file(tmp_path):
    p = tmp_path / "run.stderr.log"
    p.write_text(
        "real 12.34\n"
        "user 11.00\n"
        "sys 1.00\n"
        "123456789 peak memory footprint\n",
        encoding="utf-8",
    )
    real, peak = _tool._parse_time_file(p)
    assert real == pytest.approx(12.34, abs=1e-12)
    assert peak == pytest.approx(123456789.0, abs=1e-12)


def test_write_epistasis_degree_table_signal(tmp_path):
    out = tmp_path / "degree.tsv"
    rng = np.random.default_rng(123)
    _tool._write_epistasis_degree_table(
        path=out,
        num_site=10,
        num_convergent_site=3,
        scenario="epi_signal",
        rng=rng,
    )
    df = pd.read_csv(out, sep="\t")
    assert list(df.columns) == [
        "codon_site_alignment",
        "epistasis_contact_degree_z",
        "epistasis_contact_proximity_z",
    ]
    assert df.shape[0] == 10
    assert df["codon_site_alignment"].tolist() == list(range(1, 11))
    assert abs(float(df["epistasis_contact_degree_z"].mean())) < 1e-10
    assert abs(float(df["epistasis_contact_proximity_z"].mean())) < 1e-10
