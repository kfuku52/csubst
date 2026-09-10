import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm

from csubst import genetic_code, parser_misc, scan_endpoint, substitution_scan


def context(lengths=(1., 2.), ids=(0, 1, 2)):
    q = np.array([[-1., 1, 0], [1, -2, 1], [0, 1, -1]]) * .75
    ids = np.asarray(ids)
    state = np.zeros((3, 1, 3))
    state[:, 0, 0] = 1
    state[1, 0] = [0, 0, 1]
    projection = np.eye(ids.max() + 1)[ids]
    meta = pd.DataFrame(dict(branch_id=[1, 2], parent_id=[0, 0], raw_length=lengths,
                             sn_rescaled_length=lengths, n_rescaled_length=lengths))
    g = dict(scan_rate_exposure="endpoint", scan_rate_length="raw",
             substitution_model="GY", state_cdn=state, state_nsy=state @ projection,
             instantaneous_codon_rate_matrix=q, equilibrium_frequency=np.ones(3)/3,
             iqtree_rate_values=np.ones(1))
    return g, meta, ids


@pytest.mark.parametrize("length", [0., 1e-8, .01, 1., 10., 100.])
@pytest.mark.parametrize("ids", [(0, 1, 2), (0, 0, 1)])
def test_endpoint_matches_full_codon_expm_before_grouping(length, ids):
    g, meta, ids = context((length, length), ids)
    g["state_cdn"][0, 0] = [.2, .3, .5]
    ctx = scan_endpoint.build_context(g, meta, ids)
    for source in ([0], list(range(max(ids)+1))):
        for dest in ([max(ids)], list(range(max(ids)+1))):
            observed, missing, reasons = scan_endpoint.expected_events(ctx, g["state_cdn"], g["state_nsy"], 0, source, dest)
            p = expm(g["instantaneous_codon_rate_matrix"] * length)
            expected = sum(g["state_cdn"][0, 0, i]*p[i, j] for i in range(3) for j in range(3)
                           if ids[i] in source and ids[j] in dest and ids[i] != ids[j])
            np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-25)
            assert not missing.any()


def test_endpoint_fixes_zero_direct_q_without_multiplying_by_length_again():
    g, meta, ids = context((4/3, 8/3))
    ctx = scan_endpoint.build_context(g, meta, ids)
    kwargs = dict(candidate_events=pd.DataFrame(dict(branch_id=[1], event_pp=[1.])),
                  branch_meta=meta, state_nsy=g["state_nsy"], state_cdn=g["state_cdn"],
                  site=0, from_ids=[0], to_ids=[2], target_branch_ids=[1], rate_length="raw",
                  codon_q_matrix=g["instantaneous_codon_rate_matrix"], codon_state_ids=ids)
    old = substitution_scan._rate_summary(**kwargs, rate_exposure="q_weighted")
    new = substitution_scan._rate_summary(**kwargs, rate_exposure="endpoint", endpoint_context=ctx)
    assert old["target_exposure"] == 0
    assert old["rate_status"] == "positive_event_zero_exposure"
    assert new["target_exposure"] == pytest.approx(.1576914574755899)
    assert new["scan_exposure_units"] == "expected_endpoint_events"
    assert np.isnan(new["target_exposure_branch_length"])
    assert np.isfinite(new["p_rate_enrichment_asymptotic"])


def test_zero_and_missing_reasons_and_positive_count_conflict():
    g, meta, ids = context((0., 1.))
    ctx = scan_endpoint.build_context(g, meta, ids)
    g["state_cdn"][2] = 0
    values, missing, reasons = scan_endpoint.expected_events(ctx, g["state_cdn"], g["state_nsy"], 0, [0], [2])
    assert values.tolist() == [0., 0.]
    assert missing.tolist() == [False, True]
    assert reasons.tolist() == ["zero_model_length", "missing_state"]
    # An impossible event on ONE branch must be diagnosed even if its group's
    # total exposure is positive, rather than disappearing into the group sum.
    g["state_cdn"][2, 0, 0] = 1
    out = substitution_scan._rate_summary(
        pd.DataFrame(dict(branch_id=[1], event_pp=[1.])), meta, g["state_nsy"], 0,
        [0], [2], [1, 2], "raw", "endpoint", state_cdn=g["state_cdn"], endpoint_context=ctx)
    assert out["target_exposure"] > 0
    assert out["target_positive_event_zero_exposure_branch_count"] == 1
    assert "zero_model_length:1" in out["target_exposure_diagnostics"]
    assert out["rate_status"] == "positive_event_zero_exposure"
    assert np.isnan(out["p_rate_enrichment_asymptotic"])


def test_disconnected_and_zero_source_mass_are_distinguished():
    g, meta, ids = context()
    g["instantaneous_codon_rate_matrix"] = np.array([[-1.5, 1.5, 0], [1.5, -1.5, 0], [0, 0, 0]])
    ctx = scan_endpoint.build_context(g, meta, ids)
    _, _, reasons = scan_endpoint.expected_events(ctx, g["state_cdn"], g["state_nsy"], 0, [0], [2])
    assert reasons.tolist() == ["unreachable", "unreachable"]
    _, _, reasons = scan_endpoint.expected_events(ctx, g["state_cdn"], g["state_nsy"], 0, [2], [0])
    assert reasons.tolist() == ["zero_source_mass", "zero_source_mass"]


@pytest.mark.parametrize("key,value", [("scan_rate_length", "n_rescaled"),
                                       ("scan_rate_event_mode", "called"),
                                       ("nonsyn_recode", "3di20"),
                                       ("substitution_model", "GY+F+G4"),
                                       ("substitution_model", "ECMK07+F+R4"),
                                       ("iqtree_rate_values", [.3])])
def test_unsupported_context_is_rejected(key, value):
    g, meta, ids = context()
    g[key] = value
    with pytest.raises(ValueError):
        scan_endpoint.build_context(g, meta, ids)


@pytest.mark.parametrize("bad", ["nonfinite", "negative", "row_sum", "scale", "frequency"])
def test_invalid_q_is_rejected(bad):
    g, meta, ids = context()
    if bad == "nonfinite":
        g["instantaneous_codon_rate_matrix"][0, 1] = np.nan
    elif bad == "negative":
        g["instantaneous_codon_rate_matrix"][0, 2] = -.1
    elif bad == "row_sum":
        g["instantaneous_codon_rate_matrix"][0, 0] = 0
    elif bad == "scale":
        g["instantaneous_codon_rate_matrix"] *= 2
    else:
        g["equilibrium_frequency"] = np.array([.5, .25, .25])
    with pytest.raises(ValueError):
        scan_endpoint.build_context(g, meta, ids)


@pytest.mark.parametrize("model", ["GY", "MG", "ECMrest", "ECMK07"])
def test_actual_codon_generators_and_state_permutation(model):
    table = [(aa, c) for aa, c in genetic_code.get_codon_table(1) if aa != "*"]
    codons = np.array(sorted(c for aa, c in table))
    lookup = {c: aa for aa, c in table}
    aa = sorted(set(lookup.values()))
    ids = np.array([aa.index(lookup[c]) for c in codons])
    freq = np.ones(61)/61
    if model in ("GY", "MG"):
        q = parser_misc.get_mechanistic_instantaneous_rate_matrix(dict(
            codon_orders=codons, amino_acid_orders=aa,
            synonymous_indices={a: np.flatnonzero(ids == i) for i, a in enumerate(aa)},
            omega=.3, kappa=2 if model == "GY" else None,
            equilibrium_frequency=freq, float_type=np.float64))
    else:
        filename = "ECMrest.dat" if model == "ECMrest" else "ECMunrest.dat"
        q = parser_misc.exchangeability2Q(parser_misc.read_exchangeability_matrix("substitution_matrix/"+filename, codons), freq)
    g, meta, _ = context()
    state = np.zeros((3, 1, 61))
    state[:, 0, list(codons).index("TTT")] = 1
    g.update(state_cdn=state, state_nsy=state @ np.eye(20)[ids],
             instantaneous_codon_rate_matrix=q, equilibrium_frequency=freq, substitution_model=model)
    source, dest = [aa.index("F")], [aa.index("K")]
    ctx = scan_endpoint.build_context(g, meta, ids)
    values, _, _ = scan_endpoint.expected_events(ctx, state, g["state_nsy"], 0, source, dest)
    expected = [expm(q*t)[list(codons).index("TTT"), ids == dest[0]].sum() for t in meta.raw_length]
    np.testing.assert_allclose(values, expected, rtol=1e-12)
    assert (values > 0).all()
    if model != "ECMK07":
        assert q[list(codons).index("TTT"), ids == dest[0]].sum() == 0
    order = np.random.default_rng(7).permutation(61)
    g.update(state_cdn=state[:, :, order], instantaneous_codon_rate_matrix=q[np.ix_(order, order)])
    ctx = scan_endpoint.build_context(g, meta, ids[order])
    permuted, _, _ = scan_endpoint.expected_events(ctx, g["state_cdn"], g["state_nsy"], 0, source, dest)
    np.testing.assert_allclose(permuted, values, rtol=1e-12)
