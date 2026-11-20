"""Two-state trait simulation and conditional foreground stem sampling."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar

from csubst import ete


MAX_PROPOSALS = 10_000
_BATCH_SIZE = 64
_LOG_RATE_BOUNDS = (-18.0, 18.0)
_ULTRAMETRIC_REL_TOL = 1e-6


class SamplingError(RuntimeError):
    def __init__(self):
        self.attempts = MAX_PROPOSALS
        super().__init__(
            f"No simulated trait matched the foreground conditions in {self.attempts:,} proposals."
        )


@dataclass(frozen=True)
class TraitModel:
    parents: np.ndarray
    children: tuple[tuple[int, ...], ...]
    tips: np.ndarray
    branch_ids: np.ndarray
    observed_stems: tuple[int, ...]
    positive_tips: int
    n_tips: int
    gain_rate: float
    loss_rate: float
    log_likelihood: float
    transitions: np.ndarray

    def details(self, match_tip_count):
        return {
            "gain_rate": self.gain_rate,
            "loss_rate": self.loss_rate,
            "root_state": 0,
            "log_likelihood": self.log_likelihood,
            "tip_count": self.n_tips,
            "positive_tip_count": self.positive_tips,
            "condition_on_positive_tip_count": bool(match_tip_count),
            "conditioned_origin_count": len(self.observed_stems),
            "max_proposals_per_trial": MAX_PROPOSALS,
        }


def _topology_keys(tree, keys: dict[str | tuple[int, ...], int]) -> dict[Any, int]:
    node_keys: dict[Any, int] = {}
    for node in tree.traverse("postorder"):
        children = ete.get_children(node)
        key = tuple(sorted(node_keys[child] for child in children)) if children else str(node.name)
        node_keys[node] = keys.setdefault(key, len(keys))
    return node_keys


def _time_lengths(nodes, analysis_tree, time_tree):
    names = ete.get_leaf_names(time_tree)
    if len(set(names)) != len(names) or len(set(ete.get_leaf_names(analysis_tree))) != len(names):
        raise ValueError("--fg_trait_tree and the analysis tree must have the same unique leaf names.")
    keys: dict[str | tuple[int, ...], int] = {}
    analysis_keys = _topology_keys(analysis_tree, keys)
    time_keys = _topology_keys(time_tree, keys)
    if analysis_keys[analysis_tree] != time_keys[time_tree]:
        raise ValueError("--fg_trait_tree must match the analysis tree's rooted topology and leaf names.")
    by_key = {key: node for node, key in time_keys.items()}
    lengths = [ete.get_prop(by_key[analysis_keys[node]], "dist", None) for node in nodes[1:]]
    lengths = np.asarray([0.0, *lengths], dtype=float)
    if not np.isfinite(lengths).all() or (lengths < 0).any() or not lengths.any():
        raise ValueError("--fg_trait_tree requires explicit finite nonnegative branch times and positive total time.")
    return lengths


def _check_ultrametric(lengths, parents, tips):
    # Scale before accumulating paths so the tolerance is independent of units.
    relative_lengths = lengths / lengths.max()
    depths = np.zeros_like(relative_lengths)
    for node in range(1, len(parents)):
        depths[node] = depths[parents[node]] + relative_lengths[node]
    tip_depths = depths[tips]
    spread = float(np.ptp(tip_depths) / tip_depths.max())
    if spread > _ULTRAMETRIC_REL_TOL:
        raise ValueError(
            "--fg_trait_tree requires an ultrametric time tree with contemporaneous tips. "
            f"Relative root-to-tip spread is {spread:.4g} (maximum {_ULTRAMETRIC_REL_TOL:g})."
        )


def _transitions(lengths, gain, loss):
    change = -np.expm1(-(gain + loss) * lengths)
    p01 = (gain / (gain + loss)) * change
    p10 = (loss / (gain + loss)) * change
    return np.stack((1 - p01, p01, p10, 1 - p10), axis=-1).reshape(-1, 2, 2)


def _fit_rates(parents, lengths, tip_observations, rate_model):
    # Fit in relative time units and convert to the input units after optimization.
    scale = float(lengths.max())
    relative_lengths = lengths / scale

    def objective(log_rates):
        gain, loss = np.broadcast_to(np.exp(log_rates), (2,))
        with np.errstate(divide="ignore"):
            log_transition = np.log(_transitions(relative_lengths, gain, loss))
        partial = tip_observations.copy()
        for child in range(len(parents) - 1, 0, -1):
            parent = parents[child]
            via_child = log_transition[child] + partial[child]
            partial[parent] += np.logaddexp(via_child[:, 0], via_child[:, 1])
        return -float(partial[0, 0])

    # First bracket a maximum on the log-rate grid. The likelihood can be
    # flat at high rates once transitions on all positive-length edges mix.
    grid = np.linspace(*_LOG_RATE_BOUNDS, num=19)
    grid_scores = [objective(value) for value in grid]
    if not np.isfinite(grid_scores).any():
        raise ValueError("The observed tip states have zero likelihood on --fg_trait_tree.")
    best = int(np.argmin(grid_scores))
    er = minimize_scalar(objective, method="bounded",
                         bounds=(grid[max(0, best - 1)], grid[min(best + 1, len(grid) - 1)]))
    if rate_model == "ard":
        er_start = float(np.clip(er.x, -8.0, 8.0))
        starts = [(er_start, er_start), (er_start - 2, er_start + 2), (er_start + 2, er_start - 2)]
        results = [
            minimize(objective, start, method="L-BFGS-B", bounds=[_LOG_RATE_BOUNDS] * 2)
            for start in starts
        ]
        fit = min((result for result in results if result.success), key=lambda result: result.fun, default=None)
    else:
        fit = er if er.success else None
    if fit is None or not np.isfinite(fit.fun):
        raise ValueError("Could not fit finite binary trait transition rates on --fg_trait_tree.")
    log_rates = np.broadcast_to(fit.x, (2,))
    if np.any(log_rates <= _LOG_RATE_BOUNDS[0] + 0.01) or np.any(log_rates >= _LOG_RATE_BOUNDS[1] - 0.01):
        raise ValueError("Binary trait transition rates reached a fitting boundary; try the equal-rates (er) model or check the trait tree.")
    if objective(log_rates + 24) - fit.fun <= 1e-9:
        raise ValueError("Binary trait transition rates are unidentifiable at saturation on --fg_trait_tree.")
    gain, loss = np.exp(log_rates)
    return float(gain / scale), float(loss / scale), -float(fit.fun), _transitions(relative_lengths, gain, loss)


def _stem_flags(positive, parents, children):
    """Mark maximal all-positive clades from tip observations, for each column."""
    for node in range(len(children) - 1, -1, -1):
        if children[node]:
            positive[node] = np.logical_and.reduce(positive[list(children[node])], axis=0)
    return positive[1:] & ~positive[parents[1:]]


def fit(analysis_tree, time_tree, positive_names: set[str], rate_model: str) -> TraitModel:
    nodes = list(analysis_tree.traverse("preorder"))
    lengths = _time_lengths(nodes, analysis_tree, time_tree)
    index = {node: i for i, node in enumerate(nodes)}
    children = tuple(tuple(index[child] for child in ete.get_children(node)) for node in nodes)
    parents = np.array([0, *(index[node.up] for node in nodes[1:])], dtype=np.int64)
    tips = np.array([i for i, node_children in enumerate(children) if not node_children], dtype=np.int64)
    _check_ultrametric(lengths, parents, tips)
    tip_states = np.array([nodes[i].name in positive_names for i in tips], dtype=bool)
    positive_count = int(tip_states.sum())
    if positive_count == 0 or positive_count == len(tips):
        raise ValueError("Trait CTMC requires both foreground and background tips.")
    positive: NDArray[np.bool_] = np.zeros((len(nodes), 1), dtype=bool)
    positive[tips, 0] = tip_states
    branch_ids = np.array([ete.get_prop(node, "numerical_label") for node in nodes[1:]], dtype=np.int64)
    stems = tuple(sorted(branch_ids[_stem_flags(positive, parents, children)[:, 0]].tolist()))
    obs: NDArray[np.float64] = np.zeros((len(nodes), 2), dtype=float)
    obs[tips, 1 - tip_states.astype(int)] = -np.inf
    gain, loss, likelihood, transitions = _fit_rates(parents, lengths, obs, rate_model)
    return TraitModel(parents, children, tips, branch_ids, stems, positive_count, len(tips),
                      gain, loss, likelihood, transitions)


def sample_stems(
    model: TraitModel, eligible: NDArray[np.bool_], match_tip_count: bool, rng: np.random.Generator,
) -> tuple[tuple[int, ...], int]:
    for start in range(0, MAX_PROPOSALS, _BATCH_SIZE):
        batch_size = min(_BATCH_SIZE, MAX_PROPOSALS - start)
        random = rng.random((len(model.parents) - 1, batch_size))
        positive: NDArray[np.bool_] = np.zeros((len(model.parents), batch_size), dtype=bool)
        for node in range(1, len(model.parents)):
            parent = positive[model.parents[node]].astype(np.int8)
            positive[node] = random[node - 1] < model.transitions[node, parent, 1]
        accepted = (positive[model.tips].sum(axis=0) == model.positive_tips) if match_tip_count else np.ones(batch_size, dtype=bool)
        stems = _stem_flags(positive, model.parents, model.children)
        accepted &= (stems.sum(axis=0) == len(model.observed_stems)) & ~np.any(stems[~eligible], axis=0)
        valid = np.flatnonzero(accepted)
        if valid.size:
            draw = int(valid[0])
            return tuple(sorted(model.branch_ids[stems[:, draw]].tolist())), start + draw + 1
    raise SamplingError()
