import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest


def _import_parser_pymol_with_fake_pymol(monkeypatch, pdb_fasta, chains=None, commands=None, names=None, count_atoms=None):
    if chains is None:
        chains = []
    if commands is None:
        commands = []
    if names is None:
        names = []
    if count_atoms is None:
        count_atoms = {}

    def _record_do(command):
        commands.append(command)

    fake_cmd = types.SimpleNamespace(
        get_fastastr=lambda **kwargs: pdb_fasta,
        get_chains=lambda *args, **kwargs: list(chains),
        get_names=lambda *args, **kwargs: list(names),
        count_atoms=lambda selection: count_atoms.get(selection, 0),
        do=_record_do,
    )
    fake_pymol = types.SimpleNamespace(cmd=fake_cmd)
    monkeypatch.setitem(sys.modules, "pymol", fake_pymol)
    sys.modules.pop("csubst.parser_pymol", None)
    return importlib.import_module("csubst.parser_pymol")


def test_add_coordinate_from_user_alignment_raises_descriptive_error_on_unmappable_sequence(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    user_alignment = tmp_path / "user.fa"
    user_alignment.write_text(">x_A\nBBBB\n", encoding="utf-8")
    df = pd.DataFrame({"codon_site_alignment": [1, 2, 3, 4]})
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Unable to map --user_alignment residue"):
        parser_pymol.add_coordinate_from_user_alignment(df=df, user_alignment=str(user_alignment))


def test_add_coordinate_from_user_alignment_raises_when_sequence_names_do_not_overlap(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    user_alignment = tmp_path / "user.fa"
    user_alignment.write_text(">y_B\nAAAA\n", encoding="utf-8")
    df = pd.DataFrame({"codon_site_alignment": [1, 2, 3, 4]})
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="No sequence name overlap"):
        parser_pymol.add_coordinate_from_user_alignment(df=df, user_alignment=str(user_alignment))


def test_add_coordinate_from_user_alignment_is_case_insensitive(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAaAa\n",
    )
    user_alignment = tmp_path / "user.fa"
    user_alignment.write_text(">x_A\naaaa\n", encoding="utf-8")
    df = pd.DataFrame({"codon_site_alignment": [1, 2, 3, 4]})
    monkeypatch.chdir(tmp_path)
    out = parser_pymol.add_coordinate_from_user_alignment(df=df, user_alignment=str(user_alignment))
    assert out["codon_site_x_A"].tolist() == [1, 2, 3, 4]
    assert out["aa_x_A"].tolist() == ["A", "A", "A", "A"]


def test_add_coordinate_from_user_alignment_handles_non_default_dataframe_index(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    user_alignment = tmp_path / "user.fa"
    user_alignment.write_text(">x_A\nAAAA\n", encoding="utf-8")
    df = pd.DataFrame({"codon_site_alignment": [1, 2, 3, 4]}, index=[10, 11, 12, 13])
    monkeypatch.chdir(tmp_path)
    out = parser_pymol.add_coordinate_from_user_alignment(df=df, user_alignment=str(user_alignment))
    assert out.loc[[10, 11, 12, 13], "codon_site_x_A"].tolist() == [1, 2, 3, 4]
    assert out.loc[[10, 11, 12, 13], "aa_x_A"].tolist() == ["A", "A", "A", "A"]


def test_mask_subunit_handles_nan_identity_means_without_crashing(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=[],
    )
    mafft_add_fasta = tmp_path / "mafft_add.fa"
    mafft_add_fasta.write_text(">1abc_A\nAAAA\n>1abc_B\nAAAA\n", encoding="utf-8")
    g = {
        "mafft_add_fasta": str(mafft_add_fasta),
        "pdb": "1abc.pdb",
        "float_tol": 1e-9,
    }
    parser_pymol.mask_subunit(g)
    assert set(g["aa_identity_means"].keys()) == {"1abc_A", "1abc_B"}
    assert all(np.isnan(v) for v in g["aa_identity_means"].values())


def test_calc_aa_identity_uses_pdb_basename_prefix_matching(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    mafft_add_fasta = tmp_path / "mafft_add.fa"
    mafft_add_fasta.write_text(">1abc_A\nAAAA\n>query\nAATA\n", encoding="utf-8")
    g = {
        "mafft_add_fasta": str(mafft_add_fasta),
        "pdb": "1abc.pdb",
        "float_tol": 1e-9,
    }
    out = parser_pymol.calc_aa_identity(g)
    assert "1abc_A" in out["aa_identity_means"]
    assert out["aa_identity_means"]["1abc_A"] == pytest.approx(0.75)


def test_mask_subunit_extracts_chain_id_from_sequence_name(tmp_path, monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=[],
        commands=commands,
    )
    mafft_add_fasta = tmp_path / "mafft_add.fa"
    mafft_add_fasta.write_text(">1abc_A\nAAAA\n>1abc_B\nAAAT\n>query\nAATT\n", encoding="utf-8")
    g = {
        "mafft_add_fasta": str(mafft_add_fasta),
        "pdb": "/tmp/somewhere/1abc.pdb",
        "float_tol": 1e-9,
    }
    parser_pymol.mask_subunit(g)
    assert any(cmd == "color wheat, chain A and polymer.protein" for cmd in commands)
    assert not any("chain 1abc_A" in cmd for cmd in commands)


def test_write_mafft_alignment_rejects_empty_output(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    g = {
        "mafft_exe": "mafft",
        "mafft_op": -1,
        "mafft_ep": -1,
        "mafft_add_fasta": str(tmp_path / "add.fa"),
        "pdb": "1abc",
    }

    class _Proc:
        stdout = b""
        stderr = b""
        returncode = 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(parser_pymol.sequence, "write_alignment", lambda outfile, mode, g, leaf_only: None)
    monkeypatch.setattr(parser_pymol.subprocess, "run", lambda *args, **kwargs: _Proc())
    monkeypatch.setattr(parser_pymol.time, "sleep", lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match="File size of .* is 0"):
        parser_pymol.write_mafft_alignment(g)


def test_write_mafft_alignment_raises_on_mafft_nonzero_exit(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    g = {
        "mafft_exe": "mafft",
        "mafft_op": -1,
        "mafft_ep": -1,
        "mafft_add_fasta": str(tmp_path / "add.fa"),
        "pdb": "1abc",
    }

    class _Proc:
        stdout = b""
        stderr = b"mafft error"
        returncode = 1

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(parser_pymol.sequence, "write_alignment", lambda outfile, mode, g, leaf_only: None)
    monkeypatch.setattr(parser_pymol.subprocess, "run", lambda *args, **kwargs: _Proc())
    with pytest.raises(RuntimeError, match="MAFFT failed with exit code 1"):
        parser_pymol.write_mafft_alignment(g)


def test_write_mafft_alignment_raises_when_mapout_file_is_missing(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    g = {
        "mafft_exe": "mafft",
        "mafft_op": -1,
        "mafft_ep": -1,
        "mafft_add_fasta": str(tmp_path / "add.fa"),
        "pdb": "1abc",
    }

    class _Proc:
        stdout = b">x_A\nAAAA\n"
        stderr = b""
        returncode = 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(parser_pymol.sequence, "write_alignment", lambda outfile, mode, g, leaf_only: None)
    monkeypatch.setattr(parser_pymol.subprocess, "run", lambda *args, **kwargs: _Proc())
    monkeypatch.setattr(parser_pymol.time, "sleep", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="map output file was not generated"):
        parser_pymol.write_mafft_alignment(g)


def test_write_mafft_alignment_raises_clear_error_when_mafft_executable_missing(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    g = {
        "mafft_exe": "missing-mafft",
        "mafft_op": -1,
        "mafft_ep": -1,
        "mafft_add_fasta": str(tmp_path / "add.fa"),
        "pdb": "1abc",
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(parser_pymol.sequence, "write_alignment", lambda outfile, mode, g, leaf_only: None)

    def _raise_not_found(*args, **kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(parser_pymol.subprocess, "run", _raise_not_found)
    with pytest.raises(AssertionError, match="mafft PATH cannot be found"):
        parser_pymol.write_mafft_alignment(g)


def test_mask_subunit_skips_when_no_pdb_prefixed_sequences(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=[],
    )
    mafft_add_fasta = tmp_path / "mafft_add.fa"
    mafft_add_fasta.write_text(">query1\nAAAA\n>query2\nAAAT\n", encoding="utf-8")
    g = {
        "mafft_add_fasta": str(mafft_add_fasta),
        "pdb": "1abc.pdb",
        "float_tol": 1e-9,
    }
    out = parser_pymol.mask_subunit(g)
    assert out is None
    assert g["aa_identity_means"] == {}


def test_mask_subunit_skips_identity_scan_for_single_protein_chain(monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    parser_pymol.pymol.cmd.get_chains = lambda selection=None, *_args, **_kwargs: ["A"] if selection == "polymer.protein" else []
    monkeypatch.setattr(
        parser_pymol,
        "calc_aa_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("calc_aa_identity should not run")),
    )
    out = parser_pymol.mask_subunit({"mafft_add_fasta": "unused.fa", "pdb": "1abc.pdb"})
    assert out is None


def test_add_pdb_residue_numbering_skips_pol_conts_objects(monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    parser_pymol.pymol.cmd.get_names = lambda: ["obj", "obj_pol_conts"]
    parser_pymol.pymol.cmd.get_chains = lambda *_args, **_kwargs: ["A"]
    monkeypatch.setattr(
        parser_pymol,
        "get_residue_numberings",
        lambda: {
            "obj_A": pd.DataFrame(
                {
                    "codon_site_obj_A": [1],
                    "codon_site_pdb_obj_A": [10],
                }
            )
        },
    )
    df = pd.DataFrame({"codon_site_obj_A": [1]})
    out = parser_pymol.add_pdb_residue_numbering(df)
    assert out["codon_site_pdb_obj_A"].tolist() == [10]


def test_add_coordinate_from_mafft_map_handles_regex_characters_in_sequence_name(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    map_file = tmp_path / "tmp.csubst.pdb_seq.fa.map"
    map_file.write_text(">A[1\nA,1,1\n", encoding="utf-8")
    df = pd.DataFrame({"codon_site_alignment": [1]})
    out = parser_pymol.add_coordinate_from_mafft_map(df=df, mafft_map_file=str(map_file))
    assert out["codon_site_A[1"].tolist() == [1]
    assert out["aa_A[1"].tolist() == ["A"]


def test_add_coordinate_from_mafft_map_empty_entry_keeps_aa_column_as_empty_string(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    map_file = tmp_path / "tmp.csubst.pdb_seq.fa.map"
    map_file.write_text(">A_empty\n", encoding="utf-8")
    df = pd.DataFrame({"codon_site_alignment": [1, 2]})
    out = parser_pymol.add_coordinate_from_mafft_map(df=df, mafft_map_file=str(map_file))
    assert out["codon_site_A_empty"].tolist() == [0, 0]
    assert out["aa_A_empty"].tolist() == ["", ""]


def test_add_coordinate_from_mafft_map_treats_dash_without_space_as_missing(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    map_file = tmp_path / "tmp.csubst.pdb_seq.fa.map"
    map_file.write_text(">A\nA,1,-\nC,2,2\n", encoding="utf-8")
    df = pd.DataFrame({"codon_site_alignment": [1, 2]})
    out = parser_pymol.add_coordinate_from_mafft_map(df=df, mafft_map_file=str(map_file))
    assert out["codon_site_A"].tolist() == [0, 2]
    assert out["aa_A"].tolist() == ["", "C"]


def test_add_coordinate_from_mafft_map_rejects_non_numeric_alignment_site(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    map_file = tmp_path / "tmp.csubst.pdb_seq.fa.map"
    map_file.write_text(">A\nA,1,not_a_number\n", encoding="utf-8")
    df = pd.DataFrame({"codon_site_alignment": [1]})
    with pytest.raises(ValueError, match="Invalid codon_site_alignment value"):
        parser_pymol.add_coordinate_from_mafft_map(df=df, mafft_map_file=str(map_file))


def test_add_coordinate_from_mafft_map_treats_dash_codon_site_as_missing(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    map_file = tmp_path / "tmp.csubst.pdb_seq.fa.map"
    map_file.write_text(">A\nA,-,2\n", encoding="utf-8")
    df = pd.DataFrame({"codon_site_alignment": [1, 2]})
    out = parser_pymol.add_coordinate_from_mafft_map(df=df, mafft_map_file=str(map_file))
    assert out["codon_site_A"].tolist() == [0, 0]
    assert out["aa_A"].tolist() == ["", ""]


def test_add_coordinate_from_mafft_map_rejects_non_numeric_codon_site(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    map_file = tmp_path / "tmp.csubst.pdb_seq.fa.map"
    map_file.write_text(">A\nA,not_a_number,2\n", encoding="utf-8")
    df = pd.DataFrame({"codon_site_alignment": [2]})
    with pytest.raises(ValueError, match="Invalid codon_site value"):
        parser_pymol.add_coordinate_from_mafft_map(df=df, mafft_map_file=str(map_file))


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


def test_save_6view_pdf_creates_pdf_without_nameerror(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    directions = ["pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z"]
    image_prefix = tmp_path / "tmp.csubst.pymol"
    for direction in directions:
        image_path = tmp_path / f"tmp.csubst.pymol_{direction}.png"
        parser_pymol.plt.imsave(str(image_path), np.zeros((8, 8, 3), dtype=np.float32))
    pdf_path = tmp_path / "sixview.pdf"
    parser_pymol.save_6view_pdf(
        image_prefix=str(image_prefix),
        directions=directions,
        pdf_filename=str(pdf_path),
    )
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_set_substitution_colors_lineage_uses_only_listed_branch_columns(monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=["A"],
        commands=commands,
    )
    df = pd.DataFrame(
        {
            "codon_site_pdb_obj_A": [9],
            # High probability on non-lineage branch column should be ignored.
            "N_sub_1": [0.95],
            # Listed lineage branch remains below threshold.
            "N_sub_2": [0.20],
        }
    )
    g = {
        "mode": "lineage",
        "branch_ids": np.array([2], dtype=np.int64),
        "min_single_prob": 0.8,
        "tree": None,
    }
    n_sub_cols = df.columns[df.columns.str.startswith("N_sub_")]
    parser_pymol.set_substitution_colors(df=df, g=g, object_names=["obj"], N_sub_cols=n_sub_cols)
    assert commands == []


def test_set_substitution_colors_lineage_maps_sites_to_first_qualifying_listed_branch(monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        chains=["A"],
        commands=commands,
    )
    df = pd.DataFrame(
        {
            "codon_site_pdb_obj_A": [11],
            # Column N_sub_2 is intentionally missing.
            "N_sub_3": [0.90],
        }
    )
    fake_tree = types.SimpleNamespace(traverse=lambda: [])
    g = {
        "mode": "lineage",
        "branch_ids": np.array([2, 3], dtype=np.int64),
        "min_single_prob": 0.8,
        "tree": fake_tree,
    }
    n_sub_cols = df.columns[df.columns.str.startswith("N_sub_")]
    parser_pymol.set_substitution_colors(df=df, g=g, object_names=["obj"], N_sub_cols=n_sub_cols)
    # Site should be painted with branch-3 color (red), not branch-2 color (blue).
    assert any(("0xFF0000" in cmd) and ("resi 11" in cmd) for cmd in commands)


def test_write_pymol_session_skips_ligand_preset_without_organic_atoms(tmp_path, monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">obj_A\nAAAA\n",
        chains=["A"],
        names=["obj"],
        count_atoms={"organic": 0},
        commands=commands,
    )
    monkeypatch.setattr(parser_pymol, "set_substitution_colors", lambda *args, **kwargs: None)
    parser_pymol.pymol.cmd.deselect = lambda: None
    parser_pymol.pymol.cmd.save = lambda *_args, **_kwargs: None
    df = pd.DataFrame({"codon_site_pdb_obj_A": [1], "N_sub_1": [0.9]})
    g = {
        "remove_solvent": False,
        "remove_ligand": "",
        "pymol_transparency": 0.1,
        "pymol_gray": 80,
        "pymol_surface_quality": -1,
        "mask_subunit": False,
        "session_file_path": str(tmp_path / "out.pse"),
    }
    parser_pymol.write_pymol_session(df=df, g=g)
    assert not any("preset.ligand_sites_trans_hq" in cmd for cmd in commands)
    assert not any("util.cbag organic" in cmd for cmd in commands)
    surface_quality_index = next(i for i, cmd in enumerate(commands) if "set surface_quality" in cmd)
    set_index = next(i for i, cmd in enumerate(commands) if "set transparency" in cmd)
    show_surface_index = next(i for i, cmd in enumerate(commands) if cmd == "show surface")
    assert "set surface_quality, -1" in commands[surface_quality_index]
    assert surface_quality_index < show_surface_index
    assert set_index < show_surface_index


def test_write_pymol_session_keeps_ligand_preset_with_organic_atoms(tmp_path, monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">obj_A\nAAAA\n",
        chains=["A"],
        names=["obj"],
        count_atoms={"organic": 3},
        commands=commands,
    )
    monkeypatch.setattr(parser_pymol, "set_substitution_colors", lambda *args, **kwargs: None)
    parser_pymol.pymol.cmd.deselect = lambda: None
    parser_pymol.pymol.cmd.save = lambda *_args, **_kwargs: None
    df = pd.DataFrame({"codon_site_pdb_obj_A": [1], "N_sub_1": [0.9]})
    g = {
        "remove_solvent": False,
        "remove_ligand": "",
        "pymol_transparency": 0.1,
        "pymol_gray": 80,
        "pymol_surface_quality": 0,
        "mask_subunit": False,
        "session_file_path": str(tmp_path / "out.pse"),
    }
    parser_pymol.write_pymol_session(df=df, g=g)
    assert any("preset.ligand_sites_trans_hq" in cmd for cmd in commands)
    assert any("util.cbag organic" in cmd for cmd in commands)
    assert any("set surface_quality, 0" in cmd for cmd in commands)
