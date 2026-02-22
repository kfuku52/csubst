import numpy as np
import pytest

from csubst import recoding


def _toy_grouping_g():
    amino_acids = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)
    codon_orders = np.array(["C{:02d}".format(i) for i in range(amino_acids.shape[0])], dtype=object)
    synonymous_indices = {aa: [i] for i, aa in enumerate(amino_acids.tolist())}
    matrix_groups = {aa: [codon_orders[i]] for i, aa in enumerate(amino_acids.tolist())}
    return {
        "amino_acid_orders": amino_acids,
        "codon_orders": codon_orders,
        "synonymous_indices": synonymous_indices,
        "matrix_groups": matrix_groups,
    }


def test_normalize_nonsyn_recode_accepts_aliases():
    assert recoding.normalize_nonsyn_recode("none") == "none"
    assert recoding.normalize_nonsyn_recode("dayhoff-6") == "dayhoff6"
    assert recoding.normalize_nonsyn_recode("SR_6") == "sr6"


def test_normalize_nonsyn_recode_rejects_unknown_value():
    with pytest.raises(ValueError, match="--nonsyn_recode should be one of"):
        recoding.normalize_nonsyn_recode("unknown")


def test_initialize_nonsyn_groups_none_copies_amino_acid_groups():
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "none"
    out = recoding.initialize_nonsyn_groups(g)
    assert out["nonsyn_recode"] == "none"
    assert out["nonsyn_state_orders"].tolist() == out["amino_acid_orders"].tolist()
    assert out["max_nonsynonymous_size"] == 1
    for aa in out["amino_acid_orders"]:
        assert out["nonsynonymous_indices"][aa] == out["synonymous_indices"][aa]


def test_initialize_nonsyn_groups_dayhoff6_builds_expected_membership():
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "dayhoff6"
    out = recoding.initialize_nonsyn_groups(g)
    assert out["nonsyn_state_orders"].tolist() == ["AGPST", "DENQ", "HKR", "ILMV", "FWY", "C"]
    assert out["max_nonsynonymous_size"] == 5
    for aa in list("AGPST"):
        assert out["nonsyn_aa_to_state"][aa] == "AGPST"
    for aa in list("DENQ"):
        assert out["nonsyn_aa_to_state"][aa] == "DENQ"
    expected = sorted([g["synonymous_indices"][aa][0] for aa in list("AGPST")])
    assert out["nonsynonymous_indices"]["AGPST"] == expected


def test_initialize_nonsyn_groups_requires_grouping_keys():
    with pytest.raises(ValueError, match="Missing required key"):
        recoding.initialize_nonsyn_groups({"amino_acid_orders": np.array(["A"], dtype=object)})
