"""Replay an all-branch empirical urn search and measure focal-branch influence on PEPC."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from csubst import cli, combination, empirical_urn as eu, endpoint_io, ete, main_analyze, omega, param, parser_misc, runtime

FOCAL = (1, 26, 33, 35, 40, 102)
STAT = "any2spe"


def read_fit(inputs):
    fit = inputs / "PEPC.alignment.fa"
    fit_args = [word for ext in ("state", "rate", "iqtree", "log", "treefile")
                for word in ("--iqtree_" + ext, str(fit) + "." + ext)]
    args = cli._build_parser().parse_args([
        "search", "--alignment_file", str(fit), "--rooted_tree_file", str(inputs / "PEPC.tree.nwk"),
        "--iqtree_model", "ECMK07+F+R4", "--expectation_method", "urn", "--asrv", "empirical",
        "--output_stat", STAT, "--float_digit", "12", "--b", "no", "--random_seed", "8", *fit_args,
    ])
    g = param.get_global_parameters(args)
    g = runtime.ensure_output_layout(g, create_dir=True)
    g = parser_misc.prepare_input_context(g, include_foreground=True, include_marginal=True,
                                          resolve_state_subset=True, prepare_state=True)
    endpoint_io.prepare(g)
    return g


def fit_data(g):
    n, s = (g["_endpoint_tensors"][kind] for kind in ("N", "S"))
    nb = n.shape[0]
    eligible = np.asarray(n.eligible)
    parent = np.zeros(nb, dtype=int)
    usable = np.zeros(nb, dtype=bool)
    for node in g["tree"].traverse():
        if not ete.is_root(node):
            i = int(ete.get_prop(node, "numerical_label"))
            parent[i] = int(ete.get_prop(node.up, "numerical_label"))
            usable[i] = not ete.is_root(node.up)
    ns, nt = eu._event_totals(n, routes=True)
    ss, st = eu._event_totals(s, routes=False)
    mass = ns + ss
    groups = [np.asarray(g["synonymous_indices"][a]) for a in g["amino_acid_orders"]]
    state = g["state_cdn"]
    membership = np.zeros((state.shape[2], len(groups)))
    for a, group in enumerate(groups):
        membership[group, a] = 1
    aa = (state @ membership)[parent]
    effort = np.divide(mass.sum(1), eligible.sum(1), out=np.zeros(nb), where=eligible.any(1))
    data = (eligible, parent, mass, nt, st, groups, aa, effort)
    return data, np.flatnonzero(usable & eligible.any(1))


def probabilities(g, data, train, ids):
    eligible, parent, mass, nt, st, groups, aa, effort = data
    rate, _ = eu.fit_site_rates(mass[train], eligible[train])
    exposure = np.einsum("b,bs,bsa->a", effort[train], rate * eligible[train], aa[train], optimize=True)
    count = nt[train].sum(0)
    routes = eu._route_rates(count, exposure)
    weights = (aa[ids] @ routes) * (rate * eligible[ids])[:, :, None]
    draws = eu._pool_n_totals(count, exposure, routes, weights, effort[ids])

    def include(w, d):
        return [omega.wallenius_inclusion_profiles(w[:, :, j], v, g) for j, v in enumerate(d.T) if v.any()]

    n_prob = include(weights, draws)
    support = eligible[train].sum(0)
    sr = np.divide(mass[train].sum(0) * len(train), support,
                   out=np.zeros_like(rate), where=support > 0) + 0.5
    sr = sr / sr.mean() * eligible[ids]
    s_prob = []
    state = g["state_cdn"]
    for i, group in enumerate(groups):
        if len(group) > 1:
            cdn = state[parent[ids, None, None], np.arange(state.shape[1])[None, :, None], group]
            w = (cdn @ (1. - np.eye(len(group)))) * sr[:, :, None]
            s_prob.extend(include(w, st[ids, i, :len(group)]))
    return n_prob, s_prob


def overlap(profiles, index):
    return [sum(p[index].prod(axis=0).sum() for p in kind) for kind in profiles]


def observed_counts(observed, k):
    cols = [f"branch_id_{i+1}" for i in range(k)]
    cb = pd.read_csv(observed / f"csubst_cb_{k}.tsv", sep="\t",
                     converters={"OCN" + STAT: float, "OCS" + STAT: float},
                     usecols=cols + ["OCN" + STAT, "OCS" + STAT]).set_index(cols)
    assert cb.index.is_unique
    return cb


def blind_search(g, train, all_prob, observed):
    # OCN alone admitted a superset of every candidate needed below. Candidate
    # generation and selection use no focal labels or precomputed omega values.
    combination.get_dep_ids(g)
    previous = None
    for k in range(2, 11):
        if previous is not None and len(np.unique(previous.filter(regex="^branch_id_").to_numpy())) < k:
            break
        _, combos = combination.get_node_combinations(g, arity=k, exhaustive=k == 2,
                                                       cb_passed=previous, cb_all=True, check_attr="name")
        if not len(combos):
            break
        combos = np.sort(combos, axis=1)
        cb = observed_counts(observed, k).loc[list(map(tuple, combos))].reset_index()
        use = (cb["OCN" + STAT] >= 2) & np.isin(combos, train).all(axis=1)
        cb = cb.loc[use].copy()
        ix = np.searchsorted(train, combos[use])
        for kind, profiles in zip("NS", all_prob):
            cb["EC" + kind + STAT] = sum(omega._calc_cb_site_overlap(ix, prof, np.float64) for prof in profiles)
        cb = omega.get_omega(cb, g)
        previous = main_analyze._select_high_order_candidates(cb, g)
        print(f"BLIND K={k} proposed={len(combos)} retained={len(previous)}", flush=True)
        if k >= 6:
            print(previous.to_string(index=False), flush=True)


def run(inputs, observed, output):
    g = read_fit(inputs)
    data, train = fit_data(g)
    ids = np.array(FOCAL)
    assert np.isin(ids, train).all()
    pooled = probabilities(g, data, train, train)
    blind_search(g, train, pooled, observed)
    pooled = [[profile[np.searchsorted(train, ids)] for profile in kind] for kind in pooled]
    rows = []
    for k in range(2, len(ids) + 1):
        cb = observed_counts(observed, k)
        for index in combinations(range(len(ids)), k):
            index = list(index)
            combo = ids[index]
            row = cb.loc[tuple(combo)]
            ec_all = overlap(pooled, index)
            outside = np.setdiff1d(train, combo)
            ec_out = overlap(probabilities(g, data, outside, combo), list(range(k)))
            if k == 2:
                # Check the prototype against the shipped estimator with identical folds.
                folds = combo % eu.FOLD_COUNT
                fold_train = train[~np.isin(train % eu.FOLD_COUNT, folds)]
                ref = eu.expected_counts(g, combo[None, :], g["_endpoint_tensors"]["N"],
                                         g["_endpoint_tensors"]["S"], [STAT])
                np.testing.assert_allclose(overlap(probabilities(g, data, fold_train, combo), [0, 1]),
                                           [ref["ECN" + STAT][0], ref["ECS" + STAT][0]], rtol=1e-11, atol=0)
            for method, ec in [("all", ec_all), ("exclude_combo", ec_out)]:
                entry = dict(K=k, branches=",".join(map(str, combo)), training=method,
                             training_n=len(train) if method == "all" else len(outside))
                for kind, val in zip("NS", ec):
                    entry["OC" + kind + STAT] = row["OC" + kind + STAT]
                    entry["EC" + kind + STAT] = val
                rows.append(entry)
    result = omega.get_omega(pd.DataFrame(rows), g)
    result.to_csv(output, index=False, sep="\t")

    # Hold training size fixed while retaining all six focal branches.
    others = np.setdiff1d(train, ids)
    rng = np.random.default_rng(8)
    controls = []
    for _ in range(40):
        train_control = np.setdiff1d(train, rng.choice(others, size=len(ids), replace=False))
        controls.append(overlap(probabilities(g, data, train_control, ids), np.arange(len(ids))))
    print("K=6 random six nonfocal omissions, ECN/ECS quantiles (10%,50%,90%):",
          np.quantile(controls, [.1, .5, .9], axis=0).tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "observed", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    run(a.inputs.resolve(), a.observed.resolve(), a.output.resolve())
