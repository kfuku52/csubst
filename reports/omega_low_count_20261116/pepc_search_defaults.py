"""Compare matched CSUBST branch tables with the generating null and signal expectations."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

METHODS = ("codon_model", "urn")
STEMS = (1, 5, 26, 33, 35, 40, 102, 103, 124)
COLS = ("OCNany2spe", "OCSany2spe", "ECNany2spe", "ECSany2spe", "omegaCany2spe")
SEEDS = (4099, 4241, 4381, 4519, 4657, 4793, 4931, 5077)
SCENARIOS = ("baseline", "rate_mixture", "biased")


def run_case(panel, case, work, iqtree):
    real = case == "PEPC"
    root = work / case
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    rep = panel / case
    if real:
        source = panel / "PEPC/inputs"
        fa, tree = source / "PEPC.alignment.fa", source / "PEPC.tree.nwk"
        seed = 4099
    else:
        fa, tree = rep / "sim/sim.fa", rep / "sim/sim_true_asr.treefile"
        seed = int(case.rsplit("-", 1)[1])
    fg = panel / "PEPC/inputs/foreground.txt"
    for p in (fa, tree, fg):
        shutil.copyfile(p, inputs / p.name)
    iq = root / "iqtree"
    if case.startswith("rate_mixture-"):
        iq.mkdir()
        (original,) = (rep / "ECMK07_GY0K+F/iqtree").glob("*.iqtree")
        for ext in ("iqtree", "log", "treefile", "state", "rate", "ckp.gz"):
            src = original.with_suffix("." + ext)
            shutil.copyfile(src, iq / src.name)
    args = [
        sys.executable,
        "-m",
        "csubst",
        "search",
        "--alignment_file",
        str(inputs / fa.name),
        "--rooted_tree_file",
        str(inputs / tree.name),
        "--iqtree_exe",
        iqtree,
        "--iqtree_outdir",
        str(iq),
        "--random_seed",
        str(seed),
        "--float_digit",
        "17",
        "--threads",
        "1",
        "--blas_threads",
        "1",
    ]
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MPLBACKEND="Agg",
    )
    for scope in ("all", "stem"):
        options = (
            []
            if scope == "all"
            else [
                "--output_stat",
                "any2spe",
                "--b",
                "no",
                "--foreground",
                str(inputs / fg.name),
                "--fg_exclude_wg",
                "no",
                "--exhaustive_until",
                "1",
                "--exclude_sister_pair",
                "no",
                "--max_arity",
                "6",
                "--cutoff_stat",
                "OCNany2spe,0",
            ]
        )
        for method in ("codon_model", "urn"):
            cwd = root / scope / method
            cwd.mkdir(parents=True)
            files = sorted(iq.glob("*.iqtree"))
            opts = list(options)
            if files:
                (fit,) = files
                assert "Model of substitution: ECMK07_GY0K+F+R4" in fit.read_text()
                for ext in ("iqtree", "state", "treefile", "rate", "log"):
                    opts.extend(["--iqtree_" + ext, str(fit.with_suffix("." + ext))])
            if method == "urn":
                opts += ["--expectation_method", "urn"]
            with (cwd / "search.log").open("w") as log:
                subprocess.run(
                    args + ["--outdir", "out"] + opts,
                    cwd=cwd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        print(case, scope, flush=True)


def table(out, k):
    manifest = json.loads((out / "csubst_search_run.json").read_text())
    name = f"csubst_cb_{k}.tsv"
    assert manifest["status"] == "complete" and name in manifest["outputs"]
    ids = [f"branch_id_{i + 1}" for i in range(k)]
    numeric = {x: lambda v: np.nan if v == "NA" else float(v) for x in COLS}
    df = pd.read_table(out / name, usecols=[*ids, *COLS], converters=numeric)
    df.index = df[ids].astype(str).agg(",".join, axis=1)
    assert df.index.is_unique
    return df.sort_index()


def inputs_for_scope(rep, scope):
    outputs = [rep / scope / m / "out" for m in METHODS]
    contexts = [json.loads((out / "csubst_endpoint_model.json").read_text()) for out in outputs]
    for entry in contexts:
        assert entry["substitution_posterior"] == "joint"
        assert entry["codon_parameter_precision"]["model"] == "ECMK07_GY0K+F+R4"
        assert len(entry["codon"]["rates"]) == 4
    for key in ("branch_lengths", "rates", "weights"):
        np.testing.assert_array_equal(contexts[0]["codon"][key], contexts[1]["codon"][key])
    assert contexts[0]["codon_parameter_precision"] == contexts[1]["codon_parameter_precision"]
    prov = json.loads((outputs[1] / "csubst_urn_provenance.json").read_text())
    assert (
        prov["asrv"] == "empirical_all"
        and prov["urn_model"] == "wallenius"
        and len(prov["training_branch_ids"]) == 138
    )
    assert prov["empirical_urn"]["n_draw_totals"] == "observed_branch_n_with_pooled_route_rates"
    assert prov["empirical_urn"]["s_draw_totals"] == "observed_branch_s_with_pooled_destination_rates"
    assert not (outputs[0] / "csubst_urn_provenance.json").exists()
    return outputs, prov["training_branch_ids"]


def matched(outputs, train, scope, k):
    a, b = [table(out, k) for out in outputs]
    assert a.index.equals(b.index)
    for col in COLS[:2]:
        np.testing.assert_allclose(a[col], b[col], rtol=1e-10, atol=1e-15)
    ids = [f"branch_id_{i + 1}" for i in range(k)]
    valid = a[ids].isin(train).all(axis=1)
    if scope == "all":
        for cb in (a, b):
            assert not ((cb.OCNany2spe >= 2) & (cb.omegaCany2spe >= 5) & ~valid).any()
        assert len(a) == 8446 and valid.sum() == 8308
        assert a.loc[~valid, list(COLS[2:4])].isna().all().all()
        assert b.loc[~valid, list(COLS[2:4])].isna().all().all()
    else:
        assert valid.all() and set(a.index) == {",".join(map(str, x)) for x in combinations(STEMS, k)}
    a, b = a[valid], b[valid]
    assert (a[list(COLS[2:4])] >= 0).all().all() and (b[list(COLS[2:4])] >= 0).all().all()
    return a, b


def oracle(panel, case, scope):
    if scope == "all":
        q = pd.read_table(panel / case / "sim/sim_expected_omega.tsv")
        q.index = q[["branch_id_1", "branch_id_2"]].astype(str).agg(",".join, axis=1)
    else:
        q = pd.read_table(panel / case / "truth/replay/sim_expected_omega.tsv").set_index("branches")
    assert q.index.is_unique
    return q


def score(panel, work, dest):
    dest.mkdir(parents=True, exist_ok=True)
    seeds = []
    real_summary = []
    real_tables = []
    for case in ["PEPC", *[f"{sc}-{seed}" for sc in SCENARIOS for seed in SEEDS]]:
        real = case == "PEPC"
        for scope in ("all", "stem"):
            outputs, train = inputs_for_scope(work / case, scope)
            ref = None if real else oracle(panel, case, scope)
            for k in (2,) if scope == "all" else range(2, 7):
                a, b = matched(outputs, train, scope, k)
                cutoff = [(cb.OCNany2spe >= 2) & (cb.omegaCany2spe >= 5) for cb in (a, b)]
                support = [(cb.OCNany2spe >= 1e-9) & (cb.OCSany2spe >= 1e-9) for cb in (a, b)]
                assert support[0].equals(support[1])
                use = support[0] & np.isfinite(a.omegaCany2spe) & np.isfinite(b.omegaCany2spe)
                if real:
                    z = a[["OCNany2spe", "OCSany2spe"]].copy()
                    z.insert(0, "k", k)
                    z.insert(0, "scope", scope)
                    z["omega_supported"] = use.astype(int)
                    for method, cb, hit in zip(METHODS, (a, b), cutoff):
                        for col in COLS[2:]:
                            z[method + "_" + col] = cb[col]
                        z[method + "_cutoff"] = hit.astype(int)
                        real_summary.append(
                            dict(
                                scope=scope,
                                k=k,
                                method=method,
                                pairs_or_combos=len(cb),
                                supported=use.sum(),
                                cutoff=hit.sum(),
                                cutoff_low_ocs=(hit & (cb.OCSany2spe < 1e-9)).sum(),
                                shared_cutoff=(cutoff[0] & cutoff[1]).sum(),
                                observed_n=cb.OCNany2spe.sum(),
                                observed_s=cb.OCSany2spe.sum(),
                                expected_n=cb.ECNany2spe.sum(),
                                expected_s=cb.ECSany2spe.sum(),
                                median_omega_supported=cb.omegaCany2spe[use].median()
                                if use.any()
                                else np.nan,
                            )
                        )
                    real_tables.append(z)
                    continue
                q = ref.loc[a.index]
                assert (q[[f"mu{x}any2spe_{m}" for x in "NS" for m in ["sim", "null"]]] > 0).all().all()
                target = (q.muNany2spe_sim / q.muNany2spe_null) / (q.muSany2spe_sim / q.muSany2spe_null)
                scenario, seed = case.rsplit("-", 1)
                if scenario != "biased":
                    np.testing.assert_allclose(target, 1.0, rtol=1e-12)
                for method, cb, hit in zip(METHODS, (a, b), cutoff):
                    tn, ts = q.muNany2spe_null, q.muSany2spe_null
                    en, es = cb.ECNany2spe, cb.ECSany2spe
                    raw = cb.omegaCany2spe[use]
                    true = (q.OCNany2spe_true > 0) & (q.OCSany2spe_true > 0)
                    assert ((en[true] > 0) & (es[true] > 0)).all()
                    true_omega = (q.OCNany2spe_true[true] / q.OCSany2spe_true[true]) * (es[true] / en[true])
                    seeds.append(
                        dict(
                            scenario=scenario,
                            seed=int(seed),
                            scope=scope,
                            k=k,
                            method=method,
                            n=len(cb),
                            supported=use.sum(),
                            supported_nonfinite=int(support[0].sum() - use.sum()),
                            true_supported=true.sum(),
                            cutoff=hit.sum(),
                            shared_cutoff=(cutoff[0] & cutoff[1]).sum(),
                            expected_n=en.sum(),
                            expected_s=es.sum(),
                            ref_n=tn.sum(),
                            ref_s=ts.sum(),
                            abs_n=(en - tn).abs().sum(),
                            abs_s=(es - ts).abs().sum(),
                            raw_sse=np.square(np.log1p(raw) - np.log1p(target[use])).sum(),
                            true_sse=np.square(np.log1p(true_omega) - np.log1p(target[true])).sum(),
                        )
                    )
    per_seed = pd.DataFrame(seeds)
    assert len(per_seed) == 288
    agg = per_seed.drop(columns="seed").groupby(["scenario", "scope", "k", "method"]).sum()
    for x in ("n", "s"):
        agg["wape_" + x] = agg["abs_" + x] / agg["ref_" + x]
        agg["ratio_" + x] = agg["expected_" + x] / agg["ref_" + x]
    agg["raw_log1p_rmse"] = np.sqrt(agg.raw_sse / agg.supported.replace(0, np.nan))
    agg["true_log1p_rmse"] = np.sqrt(agg.true_sse / agg.true_supported.replace(0, np.nan))
    agg.to_csv(dest / "pepc_search_defaults_simulation.tsv", sep="\t", na_rep="NA", float_format="%.12g")
    per_seed.to_csv(dest / "pepc_search_defaults_seeds.tsv", sep="\t", index=False, float_format="%.13g")
    pd.DataFrame(real_summary).to_csv(
        dest / "pepc_search_defaults_observed.tsv", sep="\t", index=False, na_rep="NA", float_format="%.12g"
    )
    pd.concat(real_tables).to_csv(
        dest / "pepc_search_defaults_branches.tsv",
        sep="\t",
        index_label="branches",
        na_rep="NA",
        float_format="%.15g",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run", help="Fit and search each alignment")
    summarize = commands.add_parser("score", help="Validate matched branches and summarize output")
    for command in (run, summarize):
        command.add_argument("panel", type=Path)
        command.add_argument("work", type=Path)
    run.add_argument("--iqtree", default="iqtree")
    run.add_argument("--jobs", default=1, type=int)
    summarize.add_argument("tables", type=Path)
    args = parser.parse_args()
    if args.action == "score":
        score(args.panel, args.work, args.tables)
    else:
        cases = ["PEPC", *[f"{sc}-{seed}" for sc in SCENARIOS for seed in SEEDS]]
        with ThreadPoolExecutor(args.jobs) as pool:
            futures = [
                pool.submit(run_case, args.panel.resolve(), case, args.work.resolve(), args.iqtree)
                for case in cases
            ]
            for future in futures:
                future.result()
