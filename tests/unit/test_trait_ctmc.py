import itertools

import numpy as np
import pytest
from scipy.linalg import expm

from csubst import ete, trait_ctmc, tree


def test_asymmetric_trait_fit_matches_enumerated_likelihood_and_time_units():
    newick = "(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1);"
    analysis = tree.add_numerical_node_labels(ete.PhyloNode(newick, format=1))
    timed = ete.PhyloNode(newick, format=1)
    model = trait_ctmc.fit(analysis, timed, {"A", "B", "E"}, "ard")
    assert model.loss_rate > model.gain_rate > 0
    assert (model.positive_tips, len(model.observed_stems)) == (3, 2)
    nodes = list(analysis.traverse("preorder"))
    states = {i: int(node.name in {"A", "B", "E"}) for i, node in enumerate(nodes) if ete.is_leaf(node)}
    internal_ids = [i for i in range(1, len(nodes)) if i not in states]
    transition = expm([[-model.gain_rate, model.gain_rate], [model.loss_rate, -model.loss_rate]])
    likelihood = 0.0
    for internal in itertools.product((0, 1), repeat=len(internal_ids)):
        state = {0: 0, **states, **dict(zip(internal_ids, internal))}
        likelihood += np.prod([transition[state[model.parents[i]], state[i]] for i in range(1, len(nodes))])
    assert model.log_likelihood == pytest.approx(np.log(likelihood))
    for node in timed.traverse():
        if not ete.is_root(node):
            node.dist *= 10
            node.children.reverse()
    scaled = trait_ctmc.fit(analysis, timed, {"A", "B", "E"}, "ard")
    assert scaled.gain_rate * 10 == pytest.approx(model.gain_rate)
    assert scaled.loss_rate * 10 == pytest.approx(model.loss_rate)
    np.testing.assert_allclose(scaled.transitions, model.transitions)


def test_trait_draws_condition_on_origins_tips_and_eligibility(monkeypatch):
    analysis = tree.add_numerical_node_labels(
        ete.PhyloNode("(((A:0.1,B:0.1)AB:1.9,(C:1.9,D:1.9)CD:0.1):1,"
                      "((E:1,F:1):1,(G:1,H:1):1):1);", format=1)
    )
    model = trait_ctmc.fit(analysis, analysis, {"A", "B", "E"}, "er")
    # Three positive tips and two origins require a cherry and a singleton.
    names_by_id = {ete.get_prop(node, "numerical_label"): node.name for node in analysis.traverse()}
    eligible = np.isin(model.branch_ids, [bid for bid, name in names_by_id.items()
                                         if name in {"AB", "CD", "A", "E"}])
    rng = np.random.default_rng(18)
    draws = [tuple(sorted(names_by_id[bid] for bid in trait_ctmc.sample_stems(model, eligible, True, rng)[0]))
             for _ in range(200)]
    assert set(draws) == {("AB", "E"), ("CD", "E"), ("A", "CD")}
    assert draws.count(("AB", "E")) > draws.count(("CD", "E"))
    monkeypatch.setattr(trait_ctmc, "MAX_PROPOSALS", 5)
    with pytest.raises(trait_ctmc.SamplingError) as err:
        trait_ctmc.sample_stems(model, np.zeros_like(eligible), True, rng)
    assert err.value.attempts == 5


@pytest.mark.parametrize("newick, message", [
    ("((A:1,C:1):1,(B:1,D:1):1);", "rooted topology"),
    ("((A:1,B):1,(C:1,D:1):1);", "explicit finite"),
    ("((A:1,B:1.01):1,(C:1,D:1):1);", "ultrametric time tree"),
    ("((A:1,B:1):1,(C:1,D:1):1);", "unidentifiable at saturation"),
])
def test_rejects_invalid_time_tree_or_unidentifiable_fit(newick, message):
    analysis = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1):1,(C:1,D:1):1);", format=1))
    with pytest.raises(ValueError, match=message):
        trait_ctmc.fit(analysis, ete.PhyloNode(newick, format=1), {"A", "C"}, "er")


@pytest.mark.parametrize("scale", [1e-9, 1e9])
def test_ultrametric_time_check_allows_roundoff_in_any_units(scale):
    newick = "(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1);"
    analysis = tree.add_numerical_node_labels(ete.PhyloNode(newick, format=1))
    timed = ete.PhyloNode(newick, format=1)
    for node in timed.traverse():
        if not ete.is_root(node):
            node.dist *= scale
        if node.name == "E":
            node.dist += 1e-7 * scale
    model = trait_ctmc.fit(analysis, timed, {"A", "B", "E"}, "er")
    assert model.gain_rate > 0
    for node in timed.traverse():
        if node.name == "E":
            node.dist += 1e-4 * scale
    with pytest.raises(ValueError, match="ultrametric time tree"):
        trait_ctmc.fit(analysis, timed, {"A", "B", "E"}, "er")
