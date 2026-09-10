from pathlib import Path

import numpy as np
import pytest

from csubst import structural_reference as sr


def chain(ids=range(1, 7)):
    observed = [(i, {"CA": np.array([i * 3.8, 0., 0.]), "N": np.array([i * 3.8 - 1.2, 1., 0.]),
                     "C": np.array([i * 3.8 + 1.2, .5, 0.])}) for i in ids]
    return dict(sequence="AAAAAA", observed=observed,
                mapping=[dict(position=i, monomer="ALA", atoms=["N", "CA", "C"]) for i in range(1, 7)])


def features(n, i=1, j=3):
    values = np.zeros((n, 10))
    values[i, 7] = 5
    values[i, 8] = np.clip(j - i, -4, 4)
    values[i, 9] = np.sign(j - i) * np.log(abs(j - i) + 1)
    return values


def test_descriptor_projection_retains_full_axis_and_masks_undefined():
    labels, mapping = sr.project_descriptors(chain(), "AAAAAA", "CCCCCC", features(6))
    assert labels == "?C????"
    assert mapping[1]["partner_position"] == 4
    assert mapping[0]["mask_reason"] == "undefined_descriptor"
    assert mapping[0]["raw_3di"] == "C"  # an alphabet letter does not imply valid geometry


def test_missing_residue_cannot_compress_partner_sequence_distance():
    labels, mapping = sr.project_descriptors(chain([1, 2, 4, 5, 6]), "AAAAA", "CCCCC", features(5))
    assert labels == "??????"
    assert mapping[2]["mask_reason"] == "missing_CA"
    assert mapping[1]["mask_reason"] == "compressed_partner_distance"


@pytest.mark.parametrize("change,reason", [("backbone", "missing_backbone"),
                                         ("modified", "modified_or_unknown_neighborhood"),
                                         ("break", "chain_break")])
def test_invalid_descriptor_neighborhoods_are_masked(change, reason):
    value = chain()
    if change == "backbone":
        del value["observed"][2][1]["N"]
    elif change == "modified":
        value["mapping"][2]["monomer"] = "MSE"
    else:
        value["observed"][2][1]["CA"][1] = 20
    labels, mapping = sr.project_descriptors(value, "AAAAAA", "CCCCCC", features(6))
    assert labels == "??????"
    assert mapping[1]["mask_reason"] == reason


def test_descriptor_output_must_match_residues_and_partner_index():
    with pytest.raises(ValueError, match="sequence"):
        sr.project_descriptors(chain(), "AACAAA", "CCCCCC", features(6))
    values = features(6)
    values[1, 9] = 99
    with pytest.raises(ValueError, match="distance"):
        sr.project_descriptors(chain(), "AAAAAA", "CCCCCC", values)


def cif_fixture(tmp_path, missing=(), predicted=False, alternate=False):
    text = "data_TEST\n_entry.id TEST\n_exptl.method 'X-RAY DIFFRACTION'\n"
    if predicted:
        text += "_ma_model_list.model_id 1\n"
    text += "loop_\n" + "\n".join("_pdbx_poly_seq_scheme." + key for key in
                                  ["asym_id", "seq_id", "mon_id", "pdb_strand_id", "auth_seq_num", "pdb_seq_num", "pdb_ins_code"]) + "\n"
    text += "".join(f"A {i} ALA Z {100+i} {100+i} .\n" for i in range(1, 7))
    text += "#\nloop_\n" + "\n".join("_atom_site." + key for key in
                                      ["label_asym_id", "pdbx_PDB_model_num", "label_seq_id", "label_comp_id",
                                       "label_atom_id", "label_alt_id", "occupancy", "Cartn_x", "Cartn_y", "Cartn_z"]) + "\n"
    for i, atoms in chain()["observed"]:
        if i in missing:
            continue
        for name, coords in atoms.items():
            if alternate and i == 2 and name == "CA":
                text += f"A 1 {i} ALA CA A .3 99 0 0\n"
                text += f"A 1 {i} ALA CA B .7 {coords[0]} 0 0\n"
            else:
                text += f"A 1 {i} ALA {name} . 1 {coords[0]} {coords[1]} {coords[2]}\n"
    path = tmp_path / "reference.cif"
    path.write_text(text)
    return path


def test_mmcif_mapping_preserves_missing_positions_author_ids_and_conformer(tmp_path):
    pytest.importorskip("gemmi")
    path = cif_fixture(tmp_path, missing=(3,), alternate=True)
    result = sr.read_reference_chain(path, "A")
    assert result["sequence"] == "AAAAAA"
    assert [item[0] for item in result["observed"]] == [1, 2, 4, 5, 6]
    assert result["mapping"][1]["alt_id"] == "B"
    assert result["mapping"][1]["auth_asym_id"] == "Z"
    assert result["mapping"][1]["auth_seq_id"] == "102"
    assert result["mapping"][2]["atoms"] == []
    pdb = tmp_path / "out.pdb"
    sr.write_backbone_pdb(result, pdb)
    ca_rows = [line for line in pdb.read_text().splitlines() if line[12:16].strip() == "CA"]
    assert [int(line[22:26]) for line in ca_rows] == [1, 2, 4, 5, 6]


def test_predicted_structures_and_unknown_chains_are_rejected(tmp_path):
    pytest.importorskip("gemmi")
    with pytest.raises(ValueError, match="experimental"):
        sr.read_reference_chain(cif_fixture(tmp_path, predicted=True), "A")
    with pytest.raises(ValueError, match="scheme"):
        sr.read_reference_chain(cif_fixture(tmp_path), "B")


def test_foldseek_version_is_checked_before_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(sr.subprocess, "check_output", lambda *a, **k: "wrong\n")
    with pytest.raises(ValueError, match="release"):
        sr.extract_reference(Path("missing.cif"), "A", "foldseek", tmp_path / "out")
    assert not (tmp_path / "out").exists()
