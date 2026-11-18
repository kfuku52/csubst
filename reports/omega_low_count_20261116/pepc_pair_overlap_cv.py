"""Predict held-out pair/destination overlaps to choose empirical urn pooling."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from csubst import empirical_urn, omega
from csubst.omega_statistics import _calc_raw_omega, _calc_raw_rate
import pepc_high_arity_accuracy as high
from pepc_urn_symmetry import REPLICATION_SEEDS, allocate, inputs, overlap, reference, summarize

from compare_empirical_urn import background_omega, prepare
from pepc_high_arity_accuracy import SCENARIOS, check_joint_oracle
from pepc_partial_pooling import dest_prob, panel_case, run_in

SEEDS = (4099, 4241, 4381, 4519, 4657, 4793, 4931, 5077)


class MarginsOracle(high.StemOracle):
    """Joint K-stem reference plus unconditional single-edge destination means."""

    def __init__(self, tree, codon_table, codon_order, trait):
        super().__init__(tree, codon_table, codon_order, trait)
        self.destinations = sorted({aa for aa, codon in codon_table if codon in self.codons}) + list(codon_order)
        self.margins = {b: np.zeros(len(self.destinations)) for b in high.STEMS}

    def _add_model_group(self, evolver, root_model, models, num_sites):
        super()._add_model_group(evolver, root_model, models, num_sites)
        # The two diagnostic cases have homogeneous branch Q, stratified by
        # site rate. The parent's marginal distribution is propagated explicitly.
        assert len(models) == 1

        def visit(node, parent_prob):
            for child in node.children:
                length = child.branch_length * evolver.scale_tree
                transition = (
                    self.identity if length == 0 else evolver._exponentiate_matrix(root_model.matrix, length)
                )
                if child.name in self.stems:
                    self.margins[self.stems[child.name]] += num_sites * (
                        parent_prob @ self._event_features(transition)
                    )
                visit(child, parent_prob @ transition)

        visit(evolver.full_tree, root_model.params["state_freqs"])

    def write(self, path, sequences=None):
        super().write(path, sequences)
        pd.DataFrame(self.margins, index=self.destinations).T.to_csv("q_margins.tsv", sep="\t")


def observed_masses(g, data, ids):
    events = {}
    groups = [i for i, group in enumerate(data.groups) if len(group) > 1]
    for kind in ("N", "S"):
        tensor = g["_endpoint_tensors"][kind]
        proj = (
            tensor.project("any2spe")[ids]
            .toarray()
            .reshape(len(ids), tensor.num_group, tensor.num_state_to, tensor.num_site)
        )
        dest = (
            proj[:, 0]
            if kind == "N"
            else np.concatenate([proj[:, i, : len(data.groups[i])] for i in groups], axis=1)
        )
        events[kind] = dest.transpose(0, 2, 1)
    np.testing.assert_allclose(events["N"].sum(axis=1), data.route_totals[ids].sum(axis=1), atol=2e-12)
    np.testing.assert_allclose(
        events["S"].sum(axis=1),
        np.concatenate([data.syn_totals[ids, i, : len(data.groups[i])] for i in groups], axis=1),
        atol=2e-12,
    )
    parent = g["state_cdn"][data.parents[ids]]
    compatible = np.concatenate(
        [parent[:, :, group] @ (1.0 - np.eye(len(group))) for group in data.groups if len(group) > 1], axis=2
    )
    names = {
        "N": list(g["amino_acid_orders"]),
        "S": [g["codon_orders"][j] for i in groups for j in data.groups[i]],
    }
    return events, compatible, names


def diagnostic(panel, case):
    scenario, seed = case.split("-")
    assert scenario == "rate_mixture"
    with patch.object(high, "StemOracle", MarginsOracle):
        high.replay(panel, scenario, int(seed))
    oracle = pd.read_table("q_margins.tsv", index_col=0)
    g, data, rates, _ = inputs(panel / case)
    ids = np.array(high.STEMS)
    events, comp, names = observed_masses(g, data, ids)
    truth = pd.read_table("replay/sim_expected_omega.tsv").set_index("branches")
    ix = np.array(list(combinations(range(len(ids)), 6)))
    keys = [",".join(map(str, x)) for x in ids[ix]]
    cb = high.read_cb(panel / case / "control/out", 6).loc[keys]
    rows = []
    for channel, profiles in [
        (
            "S",
            {
                "current": comp * (rates.s_rate * data.eligible[ids])[:, :, None],
                "shared": comp * (rates.n_rate * data.eligible[ids])[:, :, None],
            },
        ),
        ("N", {"current": (data.aa_parent[ids] @ rates.routes) * (rates.n_rate * data.eligible[ids])[:, :, None]}),
    ]:
        target = truth.loc[keys, "mu" + channel + "any2spe_null"].to_numpy()
        qcounts = oracle.loc[ids, names[channel]].to_numpy()
        focus = (["GAG"] if seed == "3271" else ["CCA", "CCT"]) if channel == "S" else []
        for rate, weights in profiles.items():
            base = (
                events[channel].sum(axis=1)
                if channel == "S"
                else empirical_urn._pool_n_totals(
                    rates.count, rates.exposure, rates.routes, weights, data.effort[ids]
                )
            )
            for allocation, counts in [
                ("current", base),
                ("q_composition", allocate(base.sum(axis=1), qcounts)),
                ("q_margins", qcounts),
            ]:
                dest = np.array(
                    [
                        omega._calc_cb_site_overlap(ix, omega.wallenius_inclusion_profiles(w, n, g), np.float64)
                        for w, n in zip(weights.transpose(2, 0, 1), counts.T)
                    ]
                )
                estimated = dest.sum(axis=0)
                if rate == allocation == "current":
                    np.testing.assert_allclose(estimated, cb["EC" + channel + "any2spe"], rtol=2e-10, atol=8e-18)
                rows.append(
                    dict(
                        seed=seed,
                        kind=channel,
                        sites=rate,
                        allocation=allocation,
                        ratio=estimated.sum() / target.sum(),
                        wape=np.abs(estimated - target).sum() / target.sum(),
                        focus_share=(
                            dest[[names[channel].index(x) for x in focus]].sum() / estimated.sum()
                            if focus
                            else np.nan
                        ),
                    )
                )
    pd.DataFrame(rows).to_csv("diagnosis.tsv", sep="\t", index=False)


# Prior mass in endpoint-count units, fixed before scoring the independent panel.
ALPHAS = np.array([0.0, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, np.inf])
SPLIT_SEED = 20261118


def channel_profiles(data, ids, rates, channel, comp, counts, train):
    eligible = data.eligible[ids]
    if channel.startswith("S"):
        site_rate = rates.n_rate if channel == "S_shared" else rates.s_rate
        profile = comp * (site_rate * eligible)[:, :, None]
        opp = profile.sum(axis=1)
        mass = counts[train].sum(axis=0)
        effort = counts[train].sum(axis=1) / eligible[train].sum(axis=1)
        exp = effort @ opp[train]
        assert ((exp > 0) | (mass == 0)).all()
        coef = np.divide(mass, exp, out=np.zeros_like(mass), where=exp > 0)
        pred = opp * coef
    else:
        profile = (data.aa_parent[ids] @ rates.routes) * (rates.n_rate * eligible)[:, :, None]
        pred = empirical_urn._pool_n_totals(rates.count, rates.exposure, rates.routes, profile, np.ones(len(ids)))
    assert (pred.sum(axis=1) > 0).all()
    return profile, pred / pred.sum(axis=1, keepdims=True), pred.sum(axis=1)


def candidate_profiles(g, weights, draws, test_sites):
    """Interpolate the same two integer Wallenius profiles as production."""
    lo = np.floor(draws).astype(int)
    frac = draws - lo
    hi = np.ceil(draws).astype(int)
    b = np.arange(weights.shape[0])
    lookup = np.zeros((hi.max() + 1, len(b), test_sites.sum()))
    for k in np.unique(np.r_[lo.ravel(), hi.ravel()]):
        if k == 0:
            continue
        used = np.any((hi == k) | (lo == k), axis=0)
        p = omega.wallenius_inclusion_profiles(weights[used], np.full(used.sum(), k), g)
        lookup[k, used] = p[:, test_sites]
    return lookup[lo, b] * (1.0 - frac[:, :, None]) + lookup[hi, b] * frac[:, :, None]


def score_fold(g, profile, mass, prior, n_per_effort, ns_mass, eligible, pairs, site_folds, anchored):
    losses = np.zeros((2 if anchored else 1, len(ALPHAS)))
    single_losses = np.zeros_like(losses)
    summed_observed = np.zeros(len(pairs))
    left, right = pairs.T
    for f in range(2):
        test = site_folds == f
        train = ~test
        n_sites = eligible[:, train].sum(axis=1)
        count = mass[:, train].sum(axis=1)
        effort = ns_mass[:, train].sum(axis=1) / n_sites
        # Neither the composition nor the magnitude of the held branches'
        # validation-site counts enter their predictions.
        totals = [count.sum(axis=1) * eligible.sum(axis=1) / n_sites]
        if anchored:
            totals.append(n_per_effort * effort)
        prob = np.stack([dest_prob(count, prior, a) for a in ALPHAS])
        draws = np.concatenate([prob * total[None, :, None] for total in totals])
        for d in range(mass.shape[2]):
            obs = mass[:, test, d]
            y = (obs @ obs.T)[left, right]
            summed_observed += y
            inc = candidate_profiles(g, profile[:, :, d], draws[:, :, d], test)
            mu = (inc @ inc.transpose(0, 2, 1))[:, left, right]
            # Poisson mean loss up to a data-only constant. The responses
            # are fractional posterior overlap masses, not independent counts.
            loss = (mu - y * np.log(mu + 1e-12)).sum(axis=1)
            losses += loss.reshape(losses.shape)
            # Matched comparator, predicting single-branch destination totals
            # on exactly the same held branches and sites.
            single = inc.sum(axis=2)
            single_losses += (single - obs.sum(axis=1) * np.log(single + 1e-12)).sum(axis=1).reshape(losses.shape)
    return losses, single_losses, summed_observed


def choose_strength(g, data, ids, events, comp, pair_cb):
    counts = {kind: ev.sum(axis=1) for kind, ev in events.items()}
    pairs = np.searchsorted(ids, pair_cb[["branch_id_1", "branch_id_2"]].to_numpy())
    assert data.eligible[ids].all()  # Complete PEPC panel for the site extrapolation.
    generator = np.random.default_rng(SPLIT_SEED)
    bfold = generator.permutation(len(ids)) % 4
    sfold = generator.permutation(comp.shape[1]) % 2
    losses = {key: np.zeros(len(ALPHAS)) for key in ("S", "S_shared", "N", "N_anchor")}
    losses.update({key + "_single": np.zeros(len(ALPHAS)) for key in list(losses)})
    n_pairs = 0
    for f in range(4):
        train, held = np.flatnonzero(bfold != f), np.flatnonzero(bfold == f)
        selected = (bfold[pairs] == f).all(axis=1)
        local = np.searchsorted(held, pairs[selected])
        np.testing.assert_array_equal(held[local], pairs[selected])
        n_pairs += len(local)
        rates = empirical_urn._fit(data, ids[train])
        for channel in ("N", "S", "S_shared"):
            kind = channel[0]
            weights, prior, pooled = channel_profiles(data, ids, rates, channel, comp, counts[kind], train)
            loss, single, observed = score_fold(
                g,
                weights[held],
                events[kind][held],
                prior[held],
                pooled[held],
                data.ns_counts[ids[held]],
                data.eligible[ids[held]],
                local,
                sfold,
                kind == "N",
            )
            np.testing.assert_allclose(
                observed, pair_cb.loc[selected, "OC" + kind + "any2spe"], rtol=1e-9, atol=3e-16
            )
            losses[channel] += loss[0]
            losses[channel + "_single"] += single[0]
            if kind == "N":
                losses["N_anchor"] += loss[1]
                losses["N_anchor_single"] += single[1]
    curves = pd.DataFrame(losses, index=pd.Index(ALPHAS, name="alpha"))
    curves.to_csv("cv.tsv", sep="\t")
    assert np.isfinite(curves).all().all()
    return curves.idxmin().to_dict(), n_pairs


def results_for_case(rep, g, sets, predictions):
    scenario, seed = rep.name.split("-")
    rows = []
    for (scope, k), (_, cb, q) in sets.items():
        key = scope, k
        tn, ts = q.muNany2spe_null.to_numpy(), q.muSany2spe_null.to_numpy()
        mn, ms = q.muNany2spe_sim.to_numpy(), q.muSany2spe_sim.to_numpy()
        target = (mn / tn) / (ms / ts)
        on, os = cb.OCNany2spe.to_numpy(), cb.OCSany2spe.to_numpy()
        obs_n, obs_s = q.OCNany2spe_true.to_numpy(), q.OCSany2spe_true.to_numpy()
        tol = g["float_tol"]
        supported = (on >= tol) & (os >= tol)
        realized = (obs_n > 0) & (obs_s > 0)
        for method, (ns, ss) in predictions.items():
            en, es = ns[key], ss[key]
            assert np.isfinite(en + es).all() and (en > 0).all() and (es > 0).all()
            raw = _calc_raw_omega(_calc_raw_rate(on, en, tol), _calc_raw_rate(os, es, tol), tol)
            if method == "current":
                np.testing.assert_allclose(en, cb.ECNany2spe, rtol=2e-10, atol=8e-18)
                np.testing.assert_allclose(es, cb.ECSany2spe, rtol=2e-10, atol=8e-18)
                np.testing.assert_allclose(raw[supported], cb.omegaCany2spe.to_numpy()[supported], rtol=1e-8)

            def logloss(est, mask):
                return np.square(np.log1p(est) - np.log1p(target[mask])).sum()

            ratio = es / en
            rows.append(
                dict(
                    scenario=scenario,
                    seed=int(seed),
                    scope=scope,
                    k=k,
                    method=method,
                    combos=len(q),
                    ref_n=tn.sum(),
                    ref_s=ts.sum(),
                    ec_n=en.sum(),
                    ec_s=es.sum(),
                    abs_n=np.abs(en - tn).sum(),
                    abs_s=np.abs(es - ts).sum(),
                    supported=supported.sum(),
                    raw_sse=logloss(raw[supported], supported),
                    true_supported=realized.sum(),
                    true_sse=logloss(obs_n[realized] / obs_s[realized] * ratio[realized], realized),
                    true_s_positive=(obs_s > 0).sum(),
                    mean_sse=logloss(mn / ms * ratio, slice(None)),
                    zero_n=0,
                    zero_s=0,
                )
            )
    pd.DataFrame(rows).to_csv("results.tsv", sep="\t", index=False, float_format="%.15g")


def score_case(rep):
    g, data, rates, _ = inputs(rep)
    ids = g["_asrv_training_ids"]
    events, comp, _ = observed_masses(g, data, ids)
    cb = high.read_cb(Path("pairs"), 2)
    cb = cb[cb.branch_id_1.isin(ids) & cb.branch_id_2.isin(ids)]
    alpha, n_pairs = choose_strength(g, data, ids, events, comp, cb)
    scenario, seed = rep.name.split("-")
    pd.DataFrame([dict(scenario=scenario, seed=seed, validation_pairs=n_pairs, **alpha)]).to_csv(
        "strengths.tsv", sep="\t", index=False
    )
    # Simulation truth is read only after tuning. The fit below uses only ASR.
    sets = reference(rep, ids)
    fitted = {}
    for channel in ("N", "S", "S_shared"):
        kind = channel[0]
        counts = events[kind].sum(axis=1)
        weights, prior, pooled = channel_profiles(data, ids, rates, channel, comp, counts, np.arange(len(ids)))
        total = counts.sum(axis=1)
        observed = counts
        fitted[channel + "_cv"] = overlap(
            g, sets, weights, total[:, None] * dest_prob(counts, prior, alpha[channel])
        )
        fitted[channel + "_single"] = (
            fitted[channel + "_cv"]
            if alpha[channel] == alpha[channel + "_single"]
            else overlap(g, sets, weights, total[:, None] * dest_prob(counts, prior, alpha[channel + "_single"]))
        )
        fitted[channel + "_full"] = overlap(g, sets, weights, total[:, None] * prior)
        if kind == "N":
            predicted_total = data.effort[ids] * pooled
            fitted["N_anchor"] = overlap(
                g, sets, weights, predicted_total[:, None] * dest_prob(counts, prior, alpha["N_anchor"])
            )
            fitted["N_current"] = overlap(g, sets, weights, predicted_total[:, None] * prior)
        else:
            fitted[channel + "_current"] = overlap(g, sets, weights, observed)
    pairs = {
        "current": ("N_current", "S_current"),
        "shared_cv": ("N_current", "S_shared_current"),
        "s_paircv": ("N_current", "S_cv"),
        "n_paircv": ("N_cv", "S_current"),
        "both_paircv": ("N_cv", "S_cv"),
        "both_anchor": ("N_anchor", "S_cv"),
        "s_paircv_shared": ("N_current", "S_shared_cv"),
        "both_paircv_shared": ("N_cv", "S_shared_cv"),
        "both_anchor_shared": ("N_anchor", "S_shared_cv"),
        "s_full": ("N_current", "S_full"),
        "s_single": ("N_current", "S_single"),
        "both_single": ("N_single", "S_single"),
        "both_full": ("N_full", "S_full"),
    }
    results_for_case(rep, g, sets, {name: (fitted[n], fitted[s]) for name, (n, s) in pairs.items()})


def score_panel(panel, work, jobs, seeds):
    cases = [panel / f"{sc}-{s}" for s in seeds for sc in SCENARIOS]
    with ThreadPoolExecutor(jobs) as pool:
        tasks = [pool.submit(run_in, work / r.name, Path(__file__).resolve(), "one", r) for r in cases]
        for task in tasks:
            task.result()
    for name in ("results", "strengths"):
        pd.concat([pd.read_table(work / r.name / f"{name}.tsv") for r in cases], ignore_index=True).to_csv(
            work / f"{name}.tsv", sep="\t", index=False, float_format="%.13g"
        )
    summarize(work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("simulate", "one", "score", "diagnose"))
    parser.add_argument("panel", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--case", choices=("rate_mixture-1931", "rate_mixture-3271"))
    parser.add_argument("--retrospective", action="store_true", help="Score the previous eight-seed panel.")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--iqtree", default="iqtree")
    args = parser.parse_args()
    panel = args.panel.resolve()
    if args.stage == "simulate":
        sources = prepare(panel, "PEPC")
        ratio = background_omega(sources)
        (panel / "background-omega.txt").write_text(f"{ratio:.17g}\n")
        check_joint_oracle()
        with ThreadPoolExecutor(args.jobs) as pool:
            jobs = [
                pool.submit(panel_case, panel, sources, ratio, args.iqtree, sc, seed)
                for seed in SEEDS
                for sc in SCENARIOS
            ]
            for job in jobs:
                job.result()
    elif args.stage == "one":
        score_case(panel)
    elif args.stage == "diagnose":
        if args.case is None:
            parser.error("diagnose requires --case")
        diagnostic(panel, args.case)
    else:
        if args.work is None:
            parser.error("score requires --work")
        score_panel(panel, args.work.resolve(), args.jobs, REPLICATION_SEEDS if args.retrospective else SEEDS)


if __name__ == "__main__":
    main()
