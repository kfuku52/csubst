#!/usr/bin/env python3
"""Nested GY+FQ simulation, refitting, ASR and complete scan calibration.

Each observed alignment gets its own fitted generating model and null reference.
All three observation methods share each alignment and IQ-TREE refit. IQ-TREE
uses a fixed single thread (no AUTO tuning); model fitting/ASR and production
scan scoring are otherwise unchanged. This harness is cross-checked against CLI.
"""
import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from csubst import cli, ete, genetic_code, parser_misc, pipeline_calibration, runtime, scan_bootstrap, scan_ctmc, scan_statistics, sequence, substitution, substitution_scan, tree  # noqa: E402

MODES = ('marginal', 'joint', 'bridge')
INPUTS = ROOT/'reports/scientific_review_20260910/scan_id5/inputs'


def fit_context(sequences, directory, seed, topology, foreground):
    directory.mkdir(parents=True)
    alignment = directory/'input.fa'
    alignment.write_text(''.join('>'+name+'\n'+value+'\n' for name,value in sequences.items()))
    prefix = directory/'fit'
    command = ['iqtree2', '-s', str(alignment), '-te', str(topology), '-m', 'GY+FQ',
               '--seqtype', 'CODON1', '-T', '1', '--ancestral', '--rate', '--redo', '-v',
               '--prefix', str(prefix), '-seed', str(seed)]
    with (directory/'process.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    argv = ['scan', '--alignment_file', str(alignment), '--rooted_tree_file', str(topology),
            '--foreground', str(foreground), '--iqtree_model', 'GY+FQ', '--threads', '1',
            '--scan_pvalue_calibration', 'none', '--scan_n_permutations', '0',
            '--scan_site_plot', 'no', '--outdir', str(directory/'analysis')]
    for suffix in ('state', 'treefile', 'rate', 'iqtree', 'log'):
        argv += ['--iqtree_'+suffix, str(prefix)+'.'+suffix]
    args = cli._build_parser(show_advanced=True).parse_args(argv)
    g = cli.get_global_parameters_or_exit(args)
    g = runtime.ensure_output_layout(g, create_dir=True)
    g['current_arity'] = 2
    g = parser_misc.prepare_input_context(g, include_foreground=True, resolve_state_subset=True, prepare_state=False)
    g = parser_misc.prep_state(g, apply_site_filtering=False)
    model = scan_bootstrap.prepare_model(g)
    for key in ("state_cdn", "state_nsy", "state_pep"):
        g[key] = np.array(g[key], copy=True)
    return g, model


def evaluate(g, model, sequences=None, signal_sites=8, truth=None):
    # Real refits use IQ-TREE's printed marginals, exactly as the baseline CLI.
    # Fixed-model comparisons recompute marginal ASR from new tip emissions.
    if sequences is None:
        state = np.asarray(g['state_cdn'])
        marginal_state = g['state_nsy']
    else:
        state = np.zeros_like(g['state_cdn'])
        lookup = {c:i for i,c in enumerate(g['codon_orders'])}
        for node in ete.iter_leaves(g['tree']):
            codons = [sequences[node.name][i:i+3] for i in range(0,3*model['sites'],3)]
            bid = int(ete.get_prop(node,'numerical_label'))
            for site,codon in enumerate(codons):
                if codon in lookup:
                    state[bid,site,lookup[codon]] = 1
        groups = substitution_scan._build_codon_state_ids(g)
        posterior, _ = scan_ctmc.infer(g['tree'], state, model['q'], model['pi'], groups, 'joint')
        # Production marginal ASR has no row at the inserted binary root.
        posterior[int(ete.get_prop(g['tree'],'numerical_label'))] = 0
        marginal_state = sequence.cdn2pep_state(posterior, g)
    result = {}
    for mode in MODES:
        context = dict(g, scan_observation=mode, scan_pvalue_calibration='none', scan_n_permutations=0)
        if mode == 'marginal':
            context.update(state_cdn=state if sequences is None else posterior,
                           state_nsy=marginal_state, state_pep=marginal_state)
            on = substitution.get_substitution_tensor(marginal_state, mode='asis', g=context)
            syn = substitution.get_substitution_tensor(context['state_cdn'], mode='syn', g=context)
            tree.rescale_branch_length(context, syn, on)
            del syn
        else:
            context.update(instantaneous_codon_rate_matrix=model['q'], equilibrium_frequency=model['pi'],
                           state_cdn=state, scan_rate_exposure='endpoint', scan_rate_length='raw')
            context, on = scan_ctmc.prepare(context)
            scan_ctmc.set_branch_length_summaries(context, on)
        frame, _ = substitution_scan._scan_substitutions_core(context, on, on)
        maximum = scan_statistics.maximum_score(frame)
        signal = scan_statistics.maximum_score(frame.loc[frame.codon_site_alignment.astype(int) <= signal_sites])
        background = scan_statistics.maximum_score(frame.loc[frame.codon_site_alignment.astype(int) > signal_sites])
        result[mode] = dict(maximum=None if maximum == -np.inf else maximum,
                            signal_maximum=None if signal == -np.inf else signal,
                            background_maximum=None if background == -np.inf else background, candidates=len(frame),
                            undefined=int((~np.isfinite(frame.score_rate_enrichment.to_numpy(dtype=float))).sum()),
                            nominal_reject=bool((frame.p_rate_enrichment_asymptotic <= .05).any()))
        if truth is not None:
            errors = []
            below_root = []
            for node in g['tree'].traverse():
                if ete.is_root(node):
                    continue
                key = '|'.join(sorted(ete.get_leaf_names(node)))
                estimate = substitution.get_branch_site_sub_counts(on, int(ete.get_prop(node,'numerical_label')))
                squared = (estimate - truth[key])**2
                errors.extend(squared)
                if not ete.is_root(node.up):
                    below_root.extend(squared)
            result[mode]['count_mse'] = float(np.mean(errors))
            result[mode]['count_mse_excluding_root_edges'] = float(np.mean(below_root))
        del on
    return result


def simulate_with_truth(model, rng, alternative=False, signal_sites=8, factor=12.):
    """Exact Gillespie simulation records every nonsynonymous jump."""
    tr = ete.PhyloNode(model['topology'],format=1)
    translation = {codon:aa for aa,codon in genetic_code.get_codon_table(1)}
    amino = np.array([translation[c] for c in model['codons']])
    changed_q = model['q'].copy()
    changed_q[:,np.isin(model['codons'], ['AAA','AAG'])] *= factor
    np.fill_diagonal(changed_q,0)
    np.fill_diagonal(changed_q,-changed_q.sum(axis=1))
    draws, sequences, truth = {}, {}, {}
    for node in tr.traverse('preorder'):
        if ete.is_root(node):
            draws[id(node)] = rng.choice(len(model['pi']),model['sites'],p=model['pi'])
        else:
            states = draws[id(node.up)].copy()
            counts = np.zeros(model['sites'])
            for site, initial in enumerate(states):
                q = changed_q if alternative and node.name in ('a','e') and site < signal_sites else model['q']
                elapsed, current = 0., initial
                while True:
                    elapsed += rng.exponential(1 / -q[current,current])
                    if elapsed >= node.dist:
                        break
                    probabilities = q[current].copy()
                    probabilities[current] = 0
                    probabilities /= probabilities.sum()
                    following = rng.choice(len(probabilities),p=probabilities)
                    counts[site] += amino[current] != amino[following]
                    current = following
                states[site] = current
            draws[id(node)] = states
            truth['|'.join(sorted(ete.get_leaf_names(node)))] = counts
        if ete.is_leaf(node):
            sequences[node.name] = ''.join(model['codons'][draws[id(node)]])
    return sequences, truth


def value(record):
    return -np.inf if record['maximum'] is None else record['maximum']


def one_dataset(task):
    index, alternative, args = task
    directory = args.outdir/('alt' if alternative else 'null')/f'{index:04d}'
    directory.mkdir(parents=True)
    start = time.perf_counter()
    with open(os.devnull,'w') as sink, contextlib.redirect_stdout(sink):
        base_model = copy.deepcopy(args.model)
        seed = int(np.random.SeedSequence([args.seed,index,int(alternative)]).generate_state(1)[0] % (2**31-2))+1
        rng = np.random.default_rng(seed)
        sequences, truth = simulate_with_truth(base_model,rng,alternative,args.signal_sites,args.factor)
        observed_g, fitted = fit_context(sequences,directory/'observed',seed,args.topology,args.foreground)
        observed = evaluate(observed_g,fitted,signal_sites=args.signal_sites,truth=truth)
        references = {label:{mode:[] for mode in MODES} for label in ('fixed','refit')}
        parameters = []
        transitions = scan_bootstrap.transition_matrices(fitted)
        for replicate in range(args.replicates):
            replicate_seed = scan_bootstrap.replicate_seed(seed,replicate)
            simulated = scan_bootstrap.simulate_alignment(fitted,np.random.default_rng(replicate_seed),transitions)
            fixed = evaluate(observed_g,fitted,simulated,signal_sites=args.signal_sites)
            refit_g, refit_model = fit_context(simulated,directory/f'rep{replicate:04d}',replicate_seed,args.topology,args.foreground)
            refit = evaluate(refit_g,refit_model,signal_sites=args.signal_sites)
            parameters.append({k:refit_model['provenance'][k] for k in ('kappa','omega')})
            for mode in MODES:
                references['fixed'][mode].append(value(fixed[mode]))
                references['refit'][mode].append(value(refit[mode]))
            # Save full fit records for first dataset of each scenario; other
            # datasets retain seeds, scores and fitted parameter summaries.
            if index != 0:
                shutil.rmtree(directory/f'rep{replicate:04d}')
            del refit_g
        pvalues = {label:{mode:scan_statistics.empirical_pvalue(value(observed[mode]),reference,args.replicates)
                         for mode,reference in refs.items()} for label,refs in references.items()}
    localized_pvalues = {scope:{label:{mode:scan_statistics.empirical_pvalue(
        -np.inf if observed[mode][scope+'_maximum'] is None else observed[mode][scope+'_maximum'],reference,args.replicates)
        for mode,reference in refs.items()} for label,refs in references.items()} for scope in ('signal','background')}
    record = dict(localized_pvalues=localized_pvalues, index=index,alternative=alternative,seed=seed,observed=observed,pvalues=pvalues,
                  references={label:{mode:[None if v==-np.inf else v for v in vals] for mode,vals in refs.items()}
                              for label,refs in references.items()},
                  fitted_parameters={k:fitted['provenance'][k] for k in ('kappa','omega')},
                  refitted_parameters=parameters,seconds=time.perf_counter()-start)
    (directory/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


def summarize(records):
    output=[]
    for alternative in (False,True):
        rows=[r for r in records if r['alternative']==alternative]
        if not rows:
            continue
        for label in ('fixed','refit'):
            for mode in MODES:
                hits=sum(r['pvalues'][label][mode] <= .05 for r in rows)
                entry = dict(count_rmse=float(np.sqrt(np.mean([r['observed'][mode]['count_mse'] for r in rows]))),
                             count_rmse_excluding_root_edges=float(np.sqrt(np.mean([r['observed'][mode]['count_mse_excluding_root_edges'] for r in rows]))),
                             alternative=alternative,calibration=label,observation=mode,n=len(rows),
                             rejections=hits,rate=hits/len(rows),ci95=list(pipeline_calibration.binomial_interval(hits,len(rows))))
                if alternative:
                    for scope in ('signal','background'):
                        local_hits = sum(r['localized_pvalues'][scope][label][mode] <= .05 for r in rows)
                        entry[scope] = dict(rejections=local_hits,rate=local_hits/len(rows),
                                            ci95=list(pipeline_calibration.binomial_interval(local_hits,len(rows))))
                output.append(entry)
    return output


def verify_pilot(directory, topology, foreground):
    checks = []
    for label in ('null', 'alt'):
        case = directory/label/'0000'
        record = json.loads((case/'result.json').read_text())
        prefix = case/'observed/fit'
        for mode in MODES:
            out = case/('cli-'+mode)
            command = [sys.executable, '-m', 'csubst', 'scan', '--alignment_file', str(case/'observed/input.fa'),
                       '--rooted_tree_file', str(topology), '--foreground', str(foreground),
                       '--iqtree_model', 'GY+FQ', '--scan_observation', mode,
                       '--scan_pvalue_calibration', 'none', '--scan_site_plot', 'no', '--threads', '1',
                       '--outdir', str(out)]
            for suffix in ('state','treefile','rate','iqtree','log'):
                command += ['--iqtree_'+suffix, str(prefix)+'.'+suffix]
            if mode != 'marginal':
                command += ['--scan_rate_exposure','endpoint','--scan_rate_length','raw']
            with (case/('cli-'+mode+'.log')).open('w') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            frame = pd.read_csv(out/'csubst_scan.tsv',sep='\t')
            maximum = scan_statistics.maximum_score(frame)
            expected = value(record['observed'][mode])
            if len(frame) != record['observed'][mode]['candidates'] or not np.isclose(maximum,expected,rtol=1e-10,atol=1e-10):
                raise AssertionError((label,mode,len(frame),maximum,record['observed'][mode]))
            checks.append(dict(scenario=label,mode=mode,candidates=len(frame),
                               maximum=None if maximum==-np.inf else maximum,agrees=True))
    (directory/'cli_verification.json').write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--verify-pilot',type=Path)
    parser.add_argument('--datasets',type=int,default=100)
    parser.add_argument('--replicates',type=int,default=39)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=19092026)
    parser.add_argument('--signal-sites',type=int,default=8)
    parser.add_argument('--factor',type=float,default=12.)
    parser.add_argument('--sites',type=int,default=80)
    args=parser.parse_args()
    args.outdir=args.outdir.resolve()
    args.outdir.mkdir(parents=True,exist_ok=True)
    args.topology=(INPUTS/'tree.nwk').resolve()
    args.foreground=(INPUTS/'foreground.tsv').resolve()
    if args.verify_pilot is not None:
        verify_pilot(args.verify_pilot,args.topology,args.foreground)
        return
    with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):
        _,args.model=fit_context(sequence.read_fasta(str(INPUTS/'input.fa')),args.outdir/'generating-fit',51,args.topology,args.foreground)
    args.model['sites']=args.sites
    args.model['missing']={name:np.zeros(args.sites,dtype=bool) for name in args.model['missing']}
    tasks=[(i,alt,args) for i in range(args.datasets) for alt in (False,True)]
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(one_dataset,tasks):
            records.append(record)
            print(len(records),'/',len(tasks),round(record['seconds'],2),flush=True)
            (args.outdir/'summary.json').write_text(json.dumps(summarize(records),indent=2)+'\n')
    metadata={k:v for k,v in vars(args).items() if k!='model'}
    metadata.update(scope='nested_fitted_GY_FQ_null_fixed_topology_and_foreground',
                    implementation='production_Q_parser_ASR_loading_scan_discovery_and_maximum_score; IQTREE_fixed_single_thread')
    (args.outdir/'metadata.json').write_text(json.dumps(metadata,default=str,indent=2).replace(str(Path.home()),'${HOME}')+'\n')


if __name__=='__main__':
    main()
