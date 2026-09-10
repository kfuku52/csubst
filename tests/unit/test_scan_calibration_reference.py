import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

from csubst import ete, tree


def _load_reference():
    path = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "scan_calibration_check.py"
    spec = importlib.util.spec_from_file_location("scan_calibration_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_asr_matches_independent_latent_state_enumeration():
    reference = _load_reference()
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:0.2,B:0.3)X:0.1,C:0.4)R;", format=1))
    nodes = list(tr.traverse())
    ids = {n.name: int(ete.get_prop(n, "numerical_label")) for n in nodes}
    observations = {"A": np.array([0]), "B": np.array([1]), "C": np.array([-1])}
    log_likelihood, posterior = reference.reference_asr(tr, observations, states=2)
    mass = np.zeros_like(posterior)
    likelihood = 0.0
    for assignment in itertools.product(range(2), repeat=len(nodes)):
        if assignment[ids["A"]] != 0 or assignment[ids["B"]] != 1:
            continue
        probability = 0.5
        for node in nodes:
            if ete.is_root(node):
                continue
            parent = int(ete.get_prop(node.up, "numerical_label"))
            child = ids[node.name]
            p_same = (1 + np.exp(-2 * node.dist)) / 2
            probability *= p_same if assignment[parent] == assignment[child] else 1 - p_same
        likelihood += probability
        for bid, state in enumerate(assignment):
            mass[bid, 0, state] += probability
    assert log_likelihood == pytest.approx(np.log(likelihood))
    np.testing.assert_allclose(posterior, mass / likelihood, atol=1e-12)


def test_calibration_intervals_include_zero_discoveries_in_denominator():
    reference = _load_reference()
    lower, upper = reference.binomial_interval(0, 1000)
    assert lower == 0
    assert upper == pytest.approx(1 - 0.025 ** (1 / 1000))
