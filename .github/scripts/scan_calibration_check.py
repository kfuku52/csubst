#!/usr/bin/env python3
"""Fitted finite-state null -> reference ASR -> CSUBST tensors -> full scan.

This small scientific check is not a codon/3Di validation or a strong-FWER
proof. Matched assignment scenarios test the conditional null. Correlated
missingness and signal mixtures are explicitly labeled sensitivity analyses.
"""

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import beta

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from csubst import __version__, ete, substitution, substitution_scan, tree  # noqa: E402


SCENARIOS = {
    "balanced": {},
    "pectinate": {"shape": "pectinate"},
    "larger_clades": {"tips": 16, "clade_size": 2},
    "branch_heterogeneity": {"heterogeneous": True},
    "random_missing": {"missing": 0.2},
    "foreground_missing": {"missing": 0.1, "foreground_missing": 0.6},
    "partial_signal": {"signal_fraction": 0.125},
}


def transition_matrix(length, states):
    decay = np.exp(-float(length) * states / (states - 1))
    return np.full((states, states), (1 - decay) / states) + np.eye(states) * decay


def reference_asr(tr, observations, states, rate=1.0):
    """Pruning likelihood and inside/outside marginal posteriors on a fixed tree."""
    nodes = list(tr.traverse("preorder"))
    ids = {node: int(ete.get_prop(node, "numerical_label")) for node in nodes}
    sites = len(next(iter(observations.values())))
    transitions = {node: transition_matrix(node.dist * rate, states) for node in nodes if not ete.is_root(node)}
    inside = {}
    messages = {}
    for node in tr.traverse("postorder"):
        if ete.is_leaf(node):
            values = np.asarray(observations[node.name])
            likelihood = np.ones((sites, states))
            known = values >= 0
            likelihood[known] = np.eye(states)[values[known]]
        else:
            likelihood = np.ones((sites, states))
            for child in ete.get_children(node):
                messages[child] = inside[child] @ transitions[child].T
                likelihood *= messages[child]
        inside[node] = likelihood
    root_likelihood = inside[tr].mean(axis=1)
    if (root_likelihood <= 0).any():
        raise ValueError("Reference pruning likelihood underflowed.")
    outside = {tr: np.full((sites, states), 1.0 / states)}
    posterior = np.zeros((max(ids.values()) + 1, sites, states))
    for node in nodes:
        joint = inside[node] * outside[node]
        posterior[ids[node]] = joint / joint.sum(axis=1, keepdims=True)
        for child in ete.get_children(node):
            remaining = outside[node].copy()
            for sibling in ete.get_children(node):
                if sibling is not child:
                    remaining *= messages[sibling]
            outside[child] = remaining @ transitions[child]
    return float(np.log(root_likelihood).sum()), posterior


def _make_tree(shape="balanced", tips=8, heterogeneous=False, **_unused):
    names = [chr(ord("A") + i) for i in range(tips)]
    counter = iter(range(tips * 2))

    def topology(values):
        if len(values) == 1:
            return values[0] + ":0.25"
        split = 1 if shape == "pectinate" else len(values) // 2
        return "({},{})n{}:0.15".format(topology(values[:split]), topology(values[split:]), next(counter))

    tr = tree.add_numerical_node_labels(ete.PhyloNode(topology(names) + ";", format=1))
    if heterogeneous:
        for node in tr.traverse():
            if not ete.is_root(node):
                node.dist *= 4 if int(ete.get_prop(node, "numerical_label")) % 2 else 0.2
    return tr


def _simulate(tr, rng, states, sites, foreground, signal_sites):
    observed = {}
    latent = {}
    for node in tr.traverse("preorder"):
        if ete.is_root(node):
            latent[node] = rng.integers(states, size=sites)
            continue
        probabilities = transition_matrix(node.dist, states)[latent[node.up]]
        if node in foreground and signal_sites:
            # A foreground-specific transition pulse is a positive-control
            # alternative, not part of the homogeneous CTMC null model.
            probabilities[:signal_sites] *= 0.1
            probabilities[:signal_sites, 1] += 0.9
        cumulative = probabilities.cumsum(axis=1)
        cumulative[:, -1] = 1.0
        latent[node] = (rng.random((sites, 1)) > cumulative).sum(axis=1)
        if ete.is_leaf(node):
            observed[node.name] = latent[node].copy()
    return observed


def run_replicate(settings, seed, sites=64, states=20, permutations=999, alpha=0.05):
    rng = np.random.default_rng(seed)
    tr = _make_tree(**settings)
    clade_size = settings.get("clade_size", 1)
    pool = [node for node in tr.traverse() if not ete.is_root(node) and len(ete.get_leaf_names(node)) == clade_size]
    chosen = rng.choice(len(pool), size=2, replace=False)
    foreground_nodes = [pool[int(i)] for i in chosen]
    foreground_leaves = [list(ete.get_leaf_names(node)) for node in foreground_nodes]
    signal_sites = int(sites * settings.get("signal_fraction", 0))
    observations = _simulate(tr, rng, states, sites, foreground_nodes, signal_sites)
    foreground_leaf_set = {name for group in foreground_leaves for name in group}
    for name, values in observations.items():
        missing = settings.get("foreground_missing", settings.get("missing", 0)) if name in foreground_leaf_set else settings.get("missing", 0)
        values[rng.random(sites) < missing] = -1
    fit = minimize_scalar(
        lambda log_rate: -reference_asr(tr, observations, states, np.exp(log_rate))[0],
        bounds=(np.log(0.05), np.log(20.0)), method="bounded",
    )
    if not fit.success:
        raise RuntimeError("Reference rate fitting failed.")
    fitted_rate = float(np.exp(fit.x))
    _, state = reference_asr(tr, observations, states, fitted_rate)
    foreground_ids = []
    for node in tr.traverse():
        bid = int(ete.get_prop(node, "numerical_label"))
        node.dist *= fitted_rate
        ete.set_prop(node, "SNdist", float(node.dist))
        ete.set_prop(node, "Ndist", float(node.dist))
        leaf_set = set(ete.get_leaf_names(node))
        flags = [leaf_set.issubset(set(group)) for group in foreground_leaves]
        for i, flag in enumerate(flags, 1):
            ete.set_prop(node, "is_lineage_fg_trait_" + str(i), flag)
        ete.set_prop(node, "is_fg_trait", any(flags))
        if any(flags) and not ete.is_root(node):
            foreground_ids.append(bid)
        if ete.is_leaf(node):
            # Match an unavailable tip state, rather than treating imputation
            # at a missing tip as an observed endpoint substitution.
            state[bid, observations[node.name] < 0] = 0
    orders = np.array(list("ACDEFGHIKLMNPQRSTVWY")[:states], dtype=object)
    g = {
        "tree": tr, "state_nsy": state, "state_pep": state,
        "fg_df": pd.DataFrame({"name": [name for group in foreground_leaves for name in group],
                               "trait": [i + 1 for i, group in enumerate(foreground_leaves) for _ in group]}),
        "fg_leaf_names": {"trait": foreground_leaves}, "fg_ids": {"trait": np.array(foreground_ids)},
        "fg_stem_only": False, "nonsyn_state_orders": orders, "amino_acid_orders": orders,
        "iqtree_rate_values": np.ones(sites), "float_tol": 1e-12, "nonsyn_recode": "no",
        "scan_unit_mode": "clade", "scan_match": "any2spe", "scan_min_event_pp": 0.5,
        "scan_min_support": "2", "scan_rate_length": "raw", "scan_rate_exposure": "state_aware",
        "scan_rate_event_mode": "posterior_sum", "scan_other_scope": "all",
        "scan_pvalue_calibration": "full_scan", "scan_n_permutations": permutations,
        "scan_permutation_seed": int(seed), "min_clade_bin_count": 1, "threads": 1,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        tensor = substitution.get_substitution_tensor(state, mode="asis", g=g)
        scan, _ = substitution_scan.scan_substitutions(g, tensor)
    diagnostic = g["scan_calibration_diagnostics"]
    rejected = scan["p_rate_enrichment_empirical_maxT"].to_numpy(dtype=float) <= alpha
    null_sites = scan["site"].to_numpy(dtype=int) >= signal_sites
    return {
        "seed": int(seed), "fitted_rate": fitted_rate, "status": diagnostic["status"],
        "candidate_count": int(len(scan)), "null_family_rejected": bool((rejected & null_sites).any()),
        "signal_detected": bool((rejected & ~null_sites).any()), "signal_sites": signal_sites,
        "sampling": diagnostic["sampling"], "space_size": diagnostic.get("space_size"),
        "missing_fraction": float(np.mean([np.mean(values < 0) for values in observations.values()])),
    }


def binomial_interval(successes, total):
    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(0.975, successes + 1, total - successes))
    return [lower, upper]


def _run_task(arguments):
    return run_replicate(*arguments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--sites", type=int, default=64)
    parser.add_argument("--states", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fwer-margin", type=float, default=0.01)
    parser.add_argument("--require-calibration-bound", action="store_true")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replicates < 1 or args.sites < 1 or not 2 <= args.states <= 20 or args.permutations < 1 or not 0 < args.alpha < 1 or args.workers < 1 or args.fwer_margin < 0:
        parser.error("Use positive counts, 2..20 states and 0 < alpha < 1.")
    names = args.scenarios.split(",")
    if any(name not in SCENARIOS for name in names):
        parser.error("Unknown scenario.")
    output = {
        "schema_version": 1, "csubst_version": __version__, "python": platform.python_version(),
        "platform": platform.system() + " " + platform.machine(),
        "model": "equal-frequency finite-state CTMC, fitted global rate on a fixed tree; independent reference ASR",
        "scope": "state-aware/raw-length scan; not codon-Q, native 3Di or a strong-FWER guarantee",
        "settings": {key: value for key, value in vars(args).items() if key != "output"}, "scenarios": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    criteria_passed = True
    for name in names:
        settings = SCENARIOS[name]
        rows = []
        scenario_index = list(SCENARIOS).index(name)
        tasks = [
            (settings, int(np.random.SeedSequence([args.seed, scenario_index, i]).generate_state(1)[0]),
             args.sites, args.states, args.permutations, args.alpha)
            for i in range(args.replicates)
        ]
        with ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else contextlib.nullcontext() as executor:
            results = executor.map(_run_task, tasks, chunksize=5) if executor else map(_run_task, tasks)
            for i, row in enumerate(results, 1):
                rows.append(row)
                if i % 25 == 0:
                    print("{}: {}/{} replicates".format(name, i, args.replicates), flush=True)
        rejections = sum(row["null_family_rejected"] for row in rows)
        unavailable = sum(row["status"] not in ("conditional_assignment", "no_observed_candidates") for row in rows)
        summary = {
            "interpretation": "sensitivity_only" if "foreground_missing" in settings or "signal_fraction" in settings else "matched_conditional_assignment_null",
            "replicates": args.replicates, "null_family_rejections": rejections,
            "rejection_fraction_all_runs": rejections / args.replicates,
            "binomial_95_interval": binomial_interval(rejections, args.replicates),
            "no_candidate_runs": sum(row["candidate_count"] == 0 for row in rows),
            "unavailable_runs": unavailable, "signal_detected_runs": sum(row["signal_detected"] for row in rows),
            "trials": rows,
        }
        if summary["interpretation"] == "matched_conditional_assignment_null":
            passed = unavailable == 0 and summary["binomial_95_interval"][1] <= args.alpha + args.fwer_margin
            summary["criterion"] = {
                "max_binomial_95_upper": args.alpha + args.fwer_margin,
                "require_no_unavailable_runs": True,
                "passed": passed,
            }
            criteria_passed &= passed
        output["scenarios"][name] = summary
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("{}: FWER {}/{}; CI {}; unavailable {}".format(name, rejections, args.replicates, summary["binomial_95_interval"], unavailable), flush=True)
    return 1 if args.require_calibration_bound and not criteria_passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
