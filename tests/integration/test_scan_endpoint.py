from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm

from csubst import genetic_code, parser_misc, scan_endpoint, scan_statistics, substitution_scan


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


@pytest.mark.parametrize("bridge", [False, True])
def test_exposure_preserves_rare_paths_on_short_branches(bridge):
    states, time = 8, 1e-8
    off = states / (2 * (states - 1))
    q = np.diag(np.full(states - 1, off), 1) + np.diag(np.full(states - 1, off), -1)
    np.fill_diagonal(q, -q.sum(axis=1))
    g, meta, _ = context((time, 0.))
    posterior = np.zeros((3, 1, states))
    posterior[:, 0, 0] = 1
    pi = np.full(states, 1 / states)
    g.update(state_cdn=posterior, state_nsy=posterior, instantaneous_codon_rate_matrix=q,
             equilibrium_frequency=pi, scan_observation="bridge" if bridge else "joint")
    ctx = scan_endpoint.build_context(g, meta, np.arange(states))
    source, target = ([states - 1], [states - 2]) if bridge else ([0], [states - 1])
    actual, missing, reasons = scan_endpoint.expected_events(ctx, posterior, posterior, 0, source, target)

    # Independently expand the exponential (or its integral) in Decimal.
    # For bridge exposure, the only counted jump starts at the far end.
    with localcontext() as ctx:
        ctx.prec = 100
        t = Decimal.from_float(time)
        matrix = [[Decimal.from_float(value) * t for value in row] for row in q]
        term = [Decimal(i == 0) for i in range(states)]
        expected = Decimal(0)
        for n in range(32):
            expected += term[-1] * (t / (n + 1) if bridge else 1)
            term = [sum(v * row[j] for v, row in zip(term, matrix)) / (n + 1) for j in range(states)]
        if bridge:
            expected *= Decimal.from_float(off)
    np.testing.assert_allclose(actual, [float(expected), 0.], rtol=1e-13, atol=0.)
    assert not missing.any()
    assert reasons.tolist() == ["ok", "zero_model_length"]


def test_exposure_cache_reuses_only_vectors_and_recomputes_assignment_summaries(monkeypatch):
    g, meta, ids = context()
    ctx = scan_endpoint.build_context(g, meta, ids)
    calls = []
    original = scan_endpoint.expected_events
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(scan_endpoint, 'expected_events', counted)
    options = dict(candidate_events=pd.DataFrame(dict(branch_id=[1], event_pp=[.5])),
                   branch_meta=meta, state_nsy=g['state_nsy'], state_cdn=g['state_cdn'], site=0,
                   from_ids=[0], to_ids=[2], rate_length='raw', rate_exposure='endpoint', endpoint_context=ctx)
    cache = substitution_scan._CandidateCache()
    first = substitution_scan._rate_summary(**options, target_branch_ids=[1], exposure_cache=cache)
    second = substitution_scan._rate_summary(**options, target_branch_ids=[2], exposure_cache=cache)
    assert len(calls) == 1
    assert first['target_event_count'] == .5 and second['target_event_count'] == 0
    assert first['target_exposure'] != second['target_exposure']
    substitution_scan._rate_summary(**options, target_branch_ids=[1], exposure_cache=substitution_scan._CandidateCache())
    assert len(calls) == 2


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


def test_fixed_score_checks_cached_conflicts_against_each_assignment():
    events = np.array([1., 1., .1])
    exposure = np.array([0., 2., 2.])
    target = np.array([False, True, False])
    conflict = np.array([0])
    for other in (np.array([True, False, True]), np.array([False, False, True])):
        full, _, _ = substitution_scan._score_rate_branches(events, exposure, target, other, 'endpoint')
        cached, _, _ = substitution_scan._score_rate_branches(
            events, exposure, target, other, 'endpoint', impossible_indices=conflict,
        )
        np.testing.assert_equal(cached, full)
        if other[0]:
            with pytest.raises(ValueError, match='Undefined rate score'):
                scan_statistics.calibration_score(cached)
        else:
            assert scan_statistics.calibration_score(cached) > 0.


@pytest.mark.parametrize('mode', ['valid', 'zero_target', 'zero_control', 'missing', 'invalid'])
def test_score_only_preserves_endpoint_validation_and_testability(mode):
    lengths = (0., 1.) if mode == 'zero_target' else (1., 0.) if mode == 'zero_control' else (1., 2.)
    g, meta, ids = context(lengths)
    ctx = scan_endpoint.build_context(g, meta, ids)
    if mode == 'missing':
        g['state_cdn'][1] = 0
    elif mode == 'invalid':
        g['state_cdn'][0, 0, 0] = -1
    options = dict(candidate_events=pd.DataFrame(dict(branch_id=[1, 2], event_pp=[.5, .25])),
                   branch_meta=meta, state_nsy=g['state_nsy'], state_cdn=g['state_cdn'],
                   site=0, from_ids=[0], to_ids=[2], target_branch_ids=[1], rate_length='raw',
                   rate_exposure='endpoint', endpoint_context=ctx)
    if mode == 'invalid':
        for score_only in (False, True):
            with pytest.raises(ValueError, match='finite and nonnegative'):
                substitution_scan._rate_summary(**options, score_only=score_only)
        return
    full = substitution_scan._rate_summary(**options)
    compact = substitution_scan._rate_summary(**options, score_only=True)
    keys = ('score_rate_enrichment', 'target_event_count', 'other_event_count',
            'target_exposure', 'other_exposure')
    np.testing.assert_equal(compact, {key: full[key] for key in keys})
    assert scan_statistics.calibration_score(compact) == scan_statistics.calibration_score(full)
    if mode != 'valid':
        assert scan_statistics.calibration_score(compact) == -np.inf


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
