from pymol_fakes import import_parser_pymol_with_fake_pymol as _import_parser_pymol_with_fake_pymol

import types

import numpy as np
import pandas as pd





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


def test_set_substitution_colors_auto_uses_continuous_vesm_colors(monkeypatch):
    commands = []
    parser_pymol = _import_parser_pymol_with_fake_pymol(
        monkeypatch=monkeypatch,
        pdb_fasta=">obj_A\nAAAA\n",
        chains=["A"],
        commands=commands,
    )
    df = pd.DataFrame(
        {
            "codon_site_pdb_obj_A": [10, 11, 12],
            "vesm_structure_llr": [-2.0, 0.0, 2.0],
        }
    )
    parser_pymol.set_substitution_colors(
        df=df,
        g={"vep_model": "vesm-35m", "pymol_color_by": "auto"},
        object_names=["obj"],
        N_sub_cols=[],
    )
    color_commands = [command for command in commands if command.startswith("color 0x")]
    assert len(color_commands) == 3
    assert any("resi 10" in command for command in color_commands)
    assert any("resi 11" in command for command in color_commands)
    assert any("resi 12" in command for command in color_commands)
    assert any(command.startswith("select vesm_scored") for command in commands)
