from pymol_fakes import import_parser_pymol_with_fake_pymol as _import_parser_pymol_with_fake_pymol

import os

import numpy as np
import pandas as pd
import pytest

from csubst import ete
from csubst import tree




def test_initialize_pymol_loads_pdb_code_from_shared_cache(tmp_path, monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
        commands=commands,
    )
    cached_path = tmp_path / "cache" / "structures" / "rcsb" / "3zgb" / "3zgb.cif"
    cached_path.parent.mkdir(parents=True)
    cached_path.write_text("data_3zgb\n#\n", encoding="utf-8")
    observed = {}

    def ensure_rcsb_structure(**kwargs):
        observed.update(kwargs)
        return str(cached_path)

    monkeypatch.setattr(parser_pymol.structure_resources, "ensure_rcsb_structure", ensure_rcsb_structure)
    parser_pymol.initialize_pymol(
        pdb_id="3ZGB",
        g={
            "resource_cache_dir": str(tmp_path / "cache"),
            "database_timeout": 17,
            "resource_lock_poll": 0.25,
            "resource_lock_timeout": 33,
        },
    )
    assert observed == {
        "pdb_id": "3ZGB",
        "cache_dir": str(tmp_path / "cache"),
        "network_timeout": 17.0,
        "poll_seconds": 0.25,
        "lock_timeout_seconds": 33.0,
    }
    assert commands == ["delete all", "load {}, 3ZGB".format(cached_path)]


def test_write_mafft_alignment_uses_input_codon_alignment_before_internal_filtering(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">1ABC_A\nAAA\n",
    )
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    alignment_path = tmp_path / "full.fa"
    alignment_path.write_text(
        ">A\nGCTGCTGCT\n>B\nGCT---GCT\n",
        encoding="utf-8",
    )
    captured = {}

    class _Proc:
        stdout = b">A\nAAA\n>B\nA-A\n>1ABC_A\nAAA\n"
        stderr = b""
        returncode = 0

    def _run(cmd, cwd, **_kwargs):
        leaf_path = os.path.join(cwd, cmd[-1])
        captured["leaf_alignment"] = open(leaf_path, encoding="utf-8").read()
        open(os.path.join(cwd, "tmp.csubst.pdb_seq.fa.map"), "w", encoding="utf-8").write(
            ">1ABC_A\nA,1,1\nA,2,2\nA,3,3\n"
        )
        return _Proc()

    g = {
        "tree": tr,
        "alignment_file": str(alignment_path),
        "codon_table": [("A", "GCT")],
        "mafft_exe": "mafft",
        "mafft_op": -1,
        "mafft_ep": -1,
        "mafft_add_fasta": str(tmp_path / "add.fa"),
        "pdb": "1ABC",
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CSUBST_RUN_TMPDIR", raising=False)
    monkeypatch.setattr(
        parser_pymol.sequence,
        "write_alignment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("filtered state alignment should not be used")
        ),
    )
    monkeypatch.setattr(parser_pymol.subprocess, "run", _run)
    parser_pymol.write_mafft_alignment(g)
    assert captured["leaf_alignment"] == ">A\nAAA\n>B\nA-A\n"


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


def test_add_coordinate_from_user_alignment_matches_first_header_token(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A structure chain description\nAAAA\n",
    )
    user_alignment = tmp_path / "user.fa"
    user_alignment.write_text(">x_A user sequence description\nAAAA\n", encoding="utf-8")
    df = pd.DataFrame({"codon_site_alignment": [1, 2, 3, 4]})
    monkeypatch.chdir(tmp_path)
    out = parser_pymol.add_coordinate_from_user_alignment(df=df, user_alignment=str(user_alignment))
    assert out["codon_site_x_A"].tolist() == [1, 2, 3, 4]


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


def test_calc_aa_identity_ignores_all_gap_comparisons(tmp_path, monkeypatch):
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">x_A\nAAAA\n",
    )
    mafft_add_fasta = tmp_path / "mafft_add.fa"
    mafft_add_fasta.write_text(
        ">1abc_A\nAAAA\n>all_gap\n----\n>query\nAATA\n",
        encoding="utf-8",
    )
    out = parser_pymol.calc_aa_identity(
        {
            "mafft_add_fasta": str(mafft_add_fasta),
            "pdb": "1abc.pdb",
        }
    )
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
