"""Independent checks for the deliberately limited known-parameter null pilot."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np

from csubst import ete


def _pilot():
    path = Path(__file__).resolve().parents[2] / 'reports/scientific_review_20260910/calibrate_site_selection.py'
    spec = importlib.util.spec_from_file_location('site_selection_null_pilot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_null_pilot_pruning_matches_complete_internal_state_enumeration():
    pilot = _pilot()
    tr, q, transition = pilot.make_model()
    posterior, ids = pilot.simulate_posterior(tr, transition, 3, np.random.default_rng(2026), False)
    nodes = list(tr.traverse())
    internals = [node for node in nodes if not ete.is_leaf(node)]
    leaves = [node for node in nodes if ete.is_leaf(node)]
    expected = np.zeros_like(posterior)
    for site in range(3):
        tip_states = {node.name: posterior[ids[node.name], site].argmax() for node in leaves}
        for states in itertools.product(range(4), repeat=len(internals)):
            assignment = dict(tip_states, **{node.name: state for node, state in zip(internals, states)})
            probability = .25
            for node in nodes:
                if not ete.is_root(node):
                    probability *= transition[assignment[node.up.name], assignment[node.name]]
            for node in nodes:
                expected[ids[node.name], site, assignment[node.name]] += probability
    expected /= expected.sum(axis=2, keepdims=True)
    np.testing.assert_allclose(posterior, expected, atol=1e-12)


def test_null_pilot_keeps_undefined_draws_and_upper_tail_ties():
    pilot = _pilot()
    assert pilot.upper_tail([None, 1., 1., 1.], None) == 1.
    assert pilot.upper_tail([1., 1., 1.], 1.) == 1.
    assert pilot.upper_tail([None, 1., 1., 1.], 2.) == .2
