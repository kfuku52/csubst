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


def _toy_auto_grouping_g():
    g = _toy_grouping_g()
    aa_orders = [str(aa) for aa in g["amino_acid_orders"].tolist()]
    aa_matrix = np.vstack(
        [
            np.arange(20, dtype=np.int16),
            np.roll(np.arange(20, dtype=np.int16), 1),
            np.roll(np.arange(20, dtype=np.int16), 5),
            np.roll(np.arange(20, dtype=np.int16), 10),
        ]
    )
    base = np.arange(1, 21, dtype=np.float64)
    fmat = np.vstack(
        [
            base / base.sum(),
            base[::-1] / base.sum(),
            np.roll(base, 5) / base.sum(),
            np.roll(base, 10) / base.sum(),
        ]
    )
    nsitev = np.array([400, 400, 400, 400], dtype=np.int64)
    fr = (fmat * nsitev[:, np.newaxis]).sum(axis=0)
    fr = fr / fr.sum()
    g["alignment_file"] = ""
    g["nonsyn_recode_seed"] = 7
    g["nonsyn_recode_random_starts"] = 24
    g["_nonsyn_recode_alignment_cache"] = {
        "alignment_file": "",
        "aa_orders": tuple(aa_orders),
        "aa_matrix": aa_matrix,
        "fmat": fmat,
        "fr": fr,
        "nsitev": nsitev,
    }
    return g


def test_normalize_nonsyn_recode_accepts_aliases():
    assert recoding.normalize_nonsyn_recode("no") == "no"
    assert recoding.normalize_nonsyn_recode("3di") == "3di20"
    assert recoding.normalize_nonsyn_recode("threedi20") == "3di20"
    assert recoding.normalize_nonsyn_recode("dayhoff-6") == "dayhoff6"
    assert recoding.normalize_nonsyn_recode("SR_6") == "sr6"
    assert recoding.normalize_nonsyn_recode("sr-chi-sq") == "srchisq6"
    assert recoding.normalize_nonsyn_recode("kgb-auto") == "kgbauto6"


def test_normalize_nonsyn_recode_rejects_unknown_value():
    with pytest.raises(ValueError, match="--nonsyn_recode should be one of"):
        recoding.normalize_nonsyn_recode("unknown")
    with pytest.raises(ValueError, match="--nonsyn_recode should be one of"):
        recoding.normalize_nonsyn_recode("none")


def test_initialize_nonsyn_groups_no_copies_amino_acid_groups():
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "no"
    out = recoding.initialize_nonsyn_groups(g)
    assert out["nonsyn_recode"] == "no"
    assert out["nonsyn_state_orders"].tolist() == out["amino_acid_orders"].tolist()
    assert out["max_nonsynonymous_size"] == 1
    for aa in out["amino_acid_orders"]:
        assert out["nonsynonymous_indices"][aa] == out["synonymous_indices"][aa]


def test_initialize_nonsyn_groups_3di20_uses_aa_group_mapping():
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "3di20"
    out = recoding.initialize_nonsyn_groups(g)
    assert out["nonsyn_recode"] == "3di20"
    assert out["nonsyn_state_orders"].tolist() == out["amino_acid_orders"].tolist()
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


@pytest.mark.parametrize("scheme_name", ["srchisq6", "kgbauto6"])
def test_initialize_nonsyn_groups_auto_recode_builds_six_states(scheme_name):
    g = _toy_auto_grouping_g()
    g["nonsyn_recode"] = scheme_name
    out = recoding.initialize_nonsyn_groups(g)
    assert len(out["nonsyn_state_orders"]) == 6
    aa_orders = [str(aa) for aa in out["amino_acid_orders"].tolist()]
    grouped = "".join([str(state) for state in out["nonsyn_state_orders"].tolist()])
    assert sorted(grouped) == sorted("".join(aa_orders))
    assert set(out["nonsyn_aa_to_state"].keys()) == set(aa_orders)


@pytest.mark.parametrize("scheme_name", ["srchisq6", "kgbauto6"])
def test_initialize_nonsyn_groups_auto_recode_threaded_matches_single_thread(scheme_name, monkeypatch):
    g1 = _toy_auto_grouping_g()
    g1["nonsyn_recode"] = scheme_name
    g1["threads"] = 1
    out1 = recoding.initialize_nonsyn_groups(g1)

    monkeypatch.setattr(recoding, "_AUTO_RECODE_MIN_TOTAL_STARTS", 1)
    monkeypatch.setattr(recoding, "_AUTO_RECODE_MIN_STARTS_PER_JOB", 1)
    g2 = _toy_auto_grouping_g()
    g2["nonsyn_recode"] = scheme_name
    g2["threads"] = 4
    out2 = recoding.initialize_nonsyn_groups(g2)

    assert out1["nonsyn_state_orders"].tolist() == out2["nonsyn_state_orders"].tolist()
    assert out1["nonsyn_recode_auto_score"] == pytest.approx(out2["nonsyn_recode_auto_score"], abs=1e-12)


def test_initialize_nonsyn_groups_auto_recode_is_deterministic():
    g1 = _toy_auto_grouping_g()
    g1["nonsyn_recode"] = "srchisq6"
    out1 = recoding.initialize_nonsyn_groups(g1)
    g2 = _toy_auto_grouping_g()
    g2["nonsyn_recode"] = "srchisq6"
    out2 = recoding.initialize_nonsyn_groups(g2)
    assert out1["nonsyn_state_orders"].tolist() == out2["nonsyn_state_orders"].tolist()


def test_random_bin_assignments_matches_sequential_generator():
    n_random = 200
    num_item = 20
    num_bin = 6
    seed = 31
    rng_batch = np.random.default_rng(seed=seed)
    out_batch = recoding._random_bin_assignments(
        num_item=num_item,
        num_bin=num_bin,
        rng=rng_batch,
        n_random=n_random,
    )
    rng_ref = np.random.default_rng(seed=seed)
    out_ref = np.vstack(
        [
            recoding._random_bin_assignment(num_item=num_item, num_bin=num_bin, rng=rng_ref)
            for _ in range(n_random)
        ]
    ).astype(np.int64, copy=False)
    assert out_batch.shape == (n_random, num_item)
    assert np.array_equal(out_batch, out_ref)


def test_resolve_auto_recode_parallel_n_jobs_uses_work_scale():
    g = {
        "threads": 8,
    }
    n1 = recoding._resolve_auto_recode_parallel_n_jobs(g=g, n_random=1000, work_scale=1)
    n2 = recoding._resolve_auto_recode_parallel_n_jobs(g=g, n_random=1000, work_scale=40)
    assert n1 == 1
    assert n2 > 1


def test_resolve_auto_recode_parallel_backend_prefers_threading():
    out = recoding._resolve_auto_recode_parallel_backend(prefer_threading=True)
    assert out == "threading"


def test_resolve_auto_recode_chunk_factor_adaptive_defaults():
    assert recoding._resolve_auto_recode_chunk_factor(total_work_units=5_000_000) == 1
    assert recoding._resolve_auto_recode_chunk_factor(total_work_units=6_500_000) == 4
    assert recoding._resolve_auto_recode_chunk_factor(total_work_units=12_000_000) == 8
