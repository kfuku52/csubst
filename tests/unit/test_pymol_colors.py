from pymol_fakes import import_parser_pymol_with_fake_pymol as _import_parser_pymol_with_fake_pymol


import numpy as np
import pandas as pd
import pytest





def test_set_color_gray_skips_chains_without_nonzero_sites(monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=["A"],
        commands=commands,
    )
    residue_numberings = {
        "obj_A": pd.DataFrame({"codon_site_pdb_obj_A": [0, 0]})
    }
    parser_pymol.set_color_gray(
        object_names=["obj"],
        residue_numberings=residue_numberings,
        gray_value=80,
    )
    assert commands == []


def test_set_substitution_colors_handles_missing_any2dif_column(monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=["A"],
        commands=commands,
    )
    df = pd.DataFrame(
        {
            "codon_site_pdb_obj_A": [5],
            "OCNany2spe": [0.1],
            "N_sub_1": [0.9],
        }
    )
    g = {
        "mode": "intersection",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "single_branch_mode": False,
    }
    n_sub_cols = df.columns[df.columns.str.startswith("N_sub_")]
    parser_pymol.set_substitution_colors(df=df, g=g, object_names=["obj"], N_sub_cols=n_sub_cols)
    assert any("resi 5" in cmd for cmd in commands)


def test_set_substitution_colors_single_branch_prefers_branch_prob_over_ocn_columns(monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=["A"],
        commands=commands,
    )
    df = pd.DataFrame(
        {
            "codon_site_pdb_obj_A": [8],
            "OCNany2spe": [0.95],
            "OCNany2dif": [0.0],
            "N_sub_1": [0.95],
        }
    )
    g = {
        "mode": "intersection",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "single_branch_mode": True,
    }
    n_sub_cols = df.columns[df.columns.str.startswith("N_sub_")]
    parser_pymol.set_substitution_colors(df=df, g=g, object_names=["obj"], N_sub_cols=n_sub_cols)
    assert any(("0x800080" in cmd) and ("resi 8" in cmd) for cmd in commands)


def test_set_substitution_colors_total_mode_keeps_mapped_site_array(monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=["A"],
        commands=commands,
    )
    df = pd.DataFrame(
        {
            "codon_site_pdb_obj_A": [5, 8],
            "N_sub_1": [0.5, 0.1],
            "N_sub_2": [0.4, 0.1],
        }
    )
    g = {"mode": "total", "min_single_prob": 0.8, "single_branch_mode": False}
    n_sub_cols = df.columns[df.columns.str.startswith("N_sub_")]
    parser_pymol.set_substitution_colors(
        df=df,
        g=g,
        object_names=["obj"],
        N_sub_cols=n_sub_cols,
    )
    assert any("resi 5" in command for command in commands)
    assert not any("resi 8" in command for command in commands)


def test_set_substitution_colors_set_mode_parses_string_booleans_safely(monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=["A"],
        commands=commands,
    )
    df = pd.DataFrame(
        {
            "codon_site_pdb_obj_A": [5, 6],
            "N_set_expr": ["False", "True"],
        }
    )
    g = {"mode": "set"}
    parser_pymol.set_substitution_colors(df=df, g=g, object_names=["obj"], N_sub_cols=pd.Index([]))
    assert any("resi 6" in cmd for cmd in commands)
    assert not any("resi 5" in cmd for cmd in commands)


def test_set_substitution_colors_handles_empty_n_sub_columns(monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=["A"],
        commands=commands,
    )
    df = pd.DataFrame(
        {
            "codon_site_pdb_obj_A": [6],
            "OCNany2spe": [0.0],
            "OCNany2dif": [0.0],
        }
    )
    g = {
        "mode": "intersection",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "single_branch_mode": False,
    }
    parser_pymol.set_substitution_colors(df=df, g=g, object_names=["obj"], N_sub_cols=pd.Index([]))
    assert commands == []


def test_set_substitution_colors_skips_non_integer_or_missing_codon_sites(monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=["A"],
        commands=commands,
    )
    df = pd.DataFrame(
        {
            "codon_site_pdb_obj_A": ["", np.nan, "4.5", "7.0"],
            "OCNany2spe": [0.9, 0.9, 0.9, 0.9],
            "N_sub_1": [0.9, 0.9, 0.9, 0.9],
        }
    )
    g = {
        "mode": "intersection",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "single_branch_mode": False,
    }
    n_sub_cols = df.columns[df.columns.str.startswith("N_sub_")]
    parser_pymol.set_substitution_colors(df=df, g=g, object_names=["obj"], N_sub_cols=n_sub_cols)
    assert any("resi 7" in cmd for cmd in commands)
    assert not any("resi 4" in cmd for cmd in commands)


def test_set_substitution_colors_lineage_respects_min_single_prob(monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=["A"],
        commands=commands,
    )
    df = pd.DataFrame(
        {
            "codon_site_pdb_obj_A": [5],
            "N_sub_1": [0.6],  # above min_combinat_prob but below min_single_prob
        }
    )
    g = {
        "mode": "lineage",
        "branch_ids": np.array([1], dtype=np.int64),
        "min_single_prob": 0.8,
        "min_combinat_prob": 0.5,
        "tree": None,
    }
    n_sub_cols = df.columns[df.columns.str.startswith("N_sub_")]
    parser_pymol.set_substitution_colors(df=df, g=g, object_names=["obj"], N_sub_cols=n_sub_cols)
    assert commands == []


def test_set_substitution_colors_lineage_accepts_scalar_branch_id(monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=["A"],
        commands=commands,
    )
    df = pd.DataFrame(
        {
            "codon_site_pdb_obj_A": [7],
            "N_sub_1": [0.9],
        }
    )
    g = {
        "mode": "lineage",
        "branch_ids": np.int64(1),
        "min_single_prob": 0.8,
        "tree": None,
    }
    n_sub_cols = df.columns[df.columns.str.startswith("N_sub_")]
    parser_pymol.set_substitution_colors(df=df, g=g, object_names=["obj"], N_sub_cols=n_sub_cols)
    assert any("resi 7" in cmd for cmd in commands)


def test_set_substitution_colors_lineage_handles_empty_branch_ids_without_crashing(monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=["A"],
        commands=commands,
    )
    df = pd.DataFrame(
        {
            "codon_site_pdb_obj_A": [5],
            "N_sub_1": [0.9],
        }
    )
    g = {
        "mode": "lineage",
        "branch_ids": np.array([], dtype=np.int64),
        "min_single_prob": 0.8,
        "tree": None,
    }
    n_sub_cols = df.columns[df.columns.str.startswith("N_sub_")]
    parser_pymol.set_substitution_colors(df=df, g=g, object_names=["obj"], N_sub_cols=n_sub_cols)
    assert commands == []


def test_set_substitution_colors_lineage_handles_none_branch_ids_without_crashing(monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=["A"],
        commands=commands,
    )
    df = pd.DataFrame(
        {
            "codon_site_pdb_obj_A": [5],
            "N_sub_1": [0.9],
        }
    )
    g = {
        "mode": "lineage",
        "branch_ids": None,
        "min_single_prob": 0.8,
        "tree": None,
    }
    n_sub_cols = df.columns[df.columns.str.startswith("N_sub_")]
    parser_pymol.set_substitution_colors(df=df, g=g, object_names=["obj"], N_sub_cols=n_sub_cols)
    assert commands == []


def test_set_substitution_colors_lineage_rejects_non_integer_branch_ids(monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=["A"],
        commands=commands,
    )
    df = pd.DataFrame(
        {
            "codon_site_pdb_obj_A": [5],
            "N_sub_1": [0.9],
        }
    )
    g = {
        "mode": "lineage",
        "branch_ids": np.array([1.5]),
        "min_single_prob": 0.8,
        "tree": None,
    }
    n_sub_cols = df.columns[df.columns.str.startswith("N_sub_")]
    with pytest.raises(ValueError, match="integer-like"):
        parser_pymol.set_substitution_colors(df=df, g=g, object_names=["obj"], N_sub_cols=n_sub_cols)
