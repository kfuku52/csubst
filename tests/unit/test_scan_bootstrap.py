import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from csubst import cli, ete, scan_bootstrap as boot


def _model():
    return dict(q=np.array([[-1., 1.], [1., -1.]]), pi=np.array([.5, .5]),
                codons=np.array(['AAA', 'AAC']), topology='(A:.2,B:.3)R;', sites=100,
                missing={'A': np.zeros(100, dtype=bool), 'B': np.zeros(100, dtype=bool)})


def test_simulator_matches_two_tip_ctmc_and_preserves_missingness():
    model = _model()
    model['sites'] = 20000
    model['missing'] = {name: np.zeros(model['sites'], dtype=bool) for name in ('A', 'B')}
    model['missing']['A'][:10] = True
    simulated = boot.simulate_alignment(model, np.random.default_rng(55))
    a, b = (np.array([simulated[name][i:i+3] for i in range(0, 3*model['sites'], 3)]) for name in ('A','B'))
    assert (a[:10] == '---').all()
    # Independently derived two-state CTMC probability across path length .5.
    assert (a[10:] != b[10:]).mean() == pytest.approx((1-np.exp(-2*.5))/2, abs=.012)
    assert (b == 'AAA').mean() == pytest.approx(.5, abs=.012)


def test_model_snapshot_uses_full_alignment_and_checks_stationarity(tmp_path, monkeypatch):
    path = tmp_path / 'input.fa'
    path.write_text('>A\nAAAAAC\n>B\nAAC---\n')
    model = _model()
    g = dict(scan_pvalue_calibration='parametric_bootstrap', iqtree_model='GY+F', scan_cli_options={},
             instantaneous_codon_rate_matrix=model['q'], equilibrium_frequency=model['pi'],
             codon_orders=model['codons'], iqtree_rate_values=np.ones(2), alignment_file=str(path),
             tree=ete.PhyloNode(model['topology'], format=1))
    report = tmp_path / 'fit.iqtree'
    report.write_text('Log-likelihood of the tree: {}\n'.format(np.log(.25*(1-np.exp(-1))) + np.log(.5)))
    g['path_iqtree_iqtree'] = str(report)
    monkeypatch.setattr(boot, '_fitted_generator', lambda config, codons: (
        config['instantaneous_codon_rate_matrix'].copy(), config['equilibrium_frequency'].copy(), {},
    ))
    snapshot = boot.prepare_model(g)
    g['instantaneous_codon_rate_matrix'][:] = 0
    assert snapshot['q'][0, 0] == -1
    assert snapshot['sites'] == 2 and snapshot['missing']['B'].tolist() == [False, True]
    g['instantaneous_codon_rate_matrix'] = model['q'] = np.array([[-1.,1.],[1.,-1.]])
    g['equilibrium_frequency'] = np.array([.8, .2])
    with pytest.raises(ValueError, match='stationary'):
        boot.prepare_model(g)
    g['equilibrium_frequency'] = np.array([.5, .5])
    report.write_text('Log-likelihood of the tree: 0\n')
    with pytest.raises(ValueError, match='does not reproduce'):
        boot.prepare_model(g)


def test_precise_generator_remaps_frequency_axis_and_uses_checkpoint(tmp_path):
    import gzip
    state = tmp_path / 'fit.state'
    with gzip.open(tmp_path/'fit.ckp.gz', 'wt') as handle:
        handle.write('ModelCodon:\n kappa: 1.2122738383\n omega: 0.8983089442\nOther:\n value: 1\n')
    log = tmp_path / 'fit.log'
    log.write_text('Empirical state frequencies: 0.6000000000 0.4000000000\n')
    g = dict(path_iqtree_state=str(state), path_iqtree_log=str(log), iqtree_model='GY+F',
             codon_orders=np.array(['AAC','AAA']), amino_acid_orders=['K','N'],
             synonymous_indices={'K':[1], 'N':[0]})
    q, pi, provenance = boot._fitted_generator(g, g['codon_orders'])
    assert pi.tolist() == [.4, .6]
    assert provenance['omega'] == .8983089442
    assert provenance['kappa'] == 1.2122738383
    assert np.allclose(pi @ q, 0)
    assert np.allclose(-pi @ np.diag(q), 1)


def test_newick_preserves_fitted_branch_precision():
    tr = ete.PhyloNode('(A:.123456789123456,B:.987654321987654)R;', format=1)
    copied = ete.PhyloNode(boot._precise_newick(tr), format=1)
    assert [n.dist for n in ete.get_children(copied)] == [n.dist for n in ete.get_children(tr)]


@pytest.mark.parametrize('setting', [dict(iqtree_model='GY+F+R4'), dict(nonsyn_recode='3di20'),
                                     dict(scan_permutation_seed=-1)])
def test_unsupported_null_is_explicitly_rejected(setting):
    g = dict(scan_pvalue_calibration='parametric_bootstrap', iqtree_model='GY+F', scan_cli_options={})
    with pytest.raises(ValueError):
        boot.validate_options(dict(g, **setting))


def test_bootstrap_same_scores_give_one_and_empty_draws_stay_in_denominator(tmp_path, monkeypatch):
    g = dict(scan_n_permutations=3, scan_permutation_seed=1, scan_cli_options={}, outdir=str(tmp_path))
    observed = pd.DataFrame({'score_rate_enrichment': [2., 3., 0.], 'p_rate_enrichment_bootstrap_maxT': [np.nan]*3})
    values = iter([2., 2., -np.inf])
    monkeypatch.setattr(boot, 'run_replicate', lambda *args: next(values))
    result = boot.calibrate(g, observed, _model())
    assert result['p_rate_enrichment_bootstrap_maxT'].tolist() == [.75, .25, 1.]
    assert g['scan_bootstrap_summary']['replicates'][-1]['empty']
    monkeypatch.setattr(boot, 'run_replicate', lambda *args: 3.)
    result = boot.calibrate(g, observed, _model())
    assert (result['p_rate_enrichment_bootstrap_maxT'] == 1).all()


def test_failed_draw_cannot_be_excluded_from_bootstrap(tmp_path, monkeypatch):
    g = dict(scan_n_permutations=3, scan_permutation_seed=1, scan_cli_options={}, outdir=str(tmp_path))
    observed = pd.DataFrame({'score_rate_enrichment': [2.], 'p_rate_enrichment_bootstrap_maxT': [.01]})
    calls = []
    def fail_second(*args):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError('ASR failure')
        return 1.
    monkeypatch.setattr(boot, 'run_replicate', fail_second)
    result = boot.calibrate(g, observed, _model())
    assert len(calls) == 2
    assert result['p_rate_enrichment_bootstrap_maxT'].isna().all()
    assert result['scan_inference_status'].iloc[0] == 'failed_null_replicates'
    summary = g['scan_bootstrap_summary']
    assert (summary['success_count'], summary['failure_count']) == (1, 1)
    assert 'ASR failure' in summary['replicates'][1]['reason']


def test_child_options_recompute_asr_and_preserve_analysis_choices(tmp_path, monkeypatch):
    rooted = tmp_path / 'tree.nwk'
    rooted.write_text(_model()['topology'])
    fg = tmp_path / 'foreground.tsv'
    fg.write_text('1\t^A$\n2\t^B$\n')
    options = dict(rooted_tree_file=str(rooted), foreground=str(fg),
                   iqtree_state='/observed/input.state', iqtree_treefile='/observed/input.treefile',
                   scan_min_support='2', nonsyn_recode='kgbauto6', drop_invariant_tip_sites='tip_invariant',
                   ml_anc=False, scan_rate_event_mode='called', scan_site_plot=True,
                   full_cds_alignment_file='')
    def fake_run(command, **kwargs):
        run_dir = Path(kwargs['cwd'])
        pd.DataFrame({'score_rate_enrichment': [2.123456789012345]}).to_csv(run_dir/'csubst_scan.tsv', sep='\t', index=False)
        parsed = cli._build_parser(show_advanced=True).parse_args(command[3:])
        assert parsed.iqtree_state == parsed.iqtree_treefile == 'infer'
        assert parsed.nonsyn_recode == 'kgbauto6'
        assert parsed.drop_invariant_tip_sites == 'tip_invariant' and parsed.scan_rate_event_mode == 'called'
        assert not parsed.scan_site_plot and parsed.scan_pvalue_calibration == 'none'
        assert parsed.random_seed == 123
        assert Path(parsed.foreground).parent == run_dir
        assert parsed.full_cds_alignment_file == ''
    monkeypatch.setattr(boot.subprocess, 'run', fake_run)
    score = boot.run_replicate(options, _model(), tmp_path/'replicate', 123, boot.transition_matrices(_model()))
    assert score == pytest.approx(2.123456789012345)
    assert 'command.json' in [p.name for p in (tmp_path/'replicate').iterdir()]
    assert options['iqtree_state'] == '/observed/input.state'


def test_capture_keeps_original_auto_choice_and_absolutizes_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path('training.fa').write_text('>A\nAAA\n')
    args = argparse.Namespace(subcommand='scan', handler=lambda: None,
                              nonsyn_recode='kgbauto6', nonsyn_recode_training_alignment='training.fa')
    options = boot.capture_options(args, cli._build_parser(show_advanced=True))
    assert options == dict(nonsyn_recode='kgbauto6', nonsyn_recode_training_alignment=str(tmp_path/'training.fa'))


def test_empty_observed_skips_calibration_but_reports_no_test(tmp_path):
    g = dict(scan_n_permutations=99, outdir=str(tmp_path))
    result = boot.calibrate(g, pd.DataFrame(), _model())
    assert result.empty and g['scan_bootstrap_summary']['status'] == 'no_observed_candidates'
    assert not list(tmp_path.iterdir())


def test_seed_is_index_based_and_manifest_serializes_empty_reference(tmp_path, monkeypatch):
    assert boot.replicate_seed(50, 2) == boot.replicate_seed(50, 2)
    assert len({boot.replicate_seed(50, i) for i in range(20)}) == 20
    g = dict(scan_n_permutations=1, scan_cli_options={}, outdir=str(tmp_path))
    monkeypatch.setattr(boot, 'run_replicate', lambda *args: -np.inf)
    result = boot.calibrate(g, pd.DataFrame({'score_rate_enrichment': [2.]}), _model())
    manifest = json.loads((Path(result.iloc[0]['scan_bootstrap_directory'])/'manifest.json').read_text())
    assert manifest['replicates'][0]['maximum_score'] is None
    assert result.iloc[0]['p_rate_enrichment_bootstrap_maxT'] == .5
