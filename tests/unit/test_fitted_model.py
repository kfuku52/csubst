import copy
import gzip

import numpy as np
import pytest

from csubst import endpoint_io, fitted_model


def context(tmp_path, scheme='F'):
    state = tmp_path/'fit.state'
    with gzip.open(tmp_path/'fit.ckp.gz', 'wt') as handle:
        handle.write('ModelCodon:\n kappa: 1.2122738383\n omega: 0.8983089442\nOther:\n value: 1\n')
    log = tmp_path/'fit.log'
    log.write_text('Empirical state frequencies: 0.6000000000 0.4000000000\n')
    return dict(path_iqtree_state=str(state), path_iqtree_log=str(log), iqtree_model='ECMK07+F+R4',
                substitution_model='GY+'+scheme, codon_orders=np.array(['AAC','AAA']),
                amino_acid_orders=['K','N'], synonymous_indices={'K':[1], 'N':[0]},
                equilibrium_frequency=np.array([.4,.6]) if scheme == 'F' else np.array([.5,.5]),
                instantaneous_codon_rate_matrix=np.array([[-1.,1.],[1.,-1.]]), omega=.8983, kappa=1.2123)


@pytest.mark.parametrize('scheme', ['F', 'FQ'])
@pytest.mark.parametrize('suffix', ['', '+G4', '+R4+I'])
def test_command_independent_precision_and_authoritative_report(tmp_path, scheme, suffix):
    g = context(tmp_path, scheme)
    g['substitution_model'] += suffix
    outputs = []
    for command in ['search', 'sites', 'scan', 'inspect', 'benchmark']:
        local = copy.deepcopy(g)
        local['subcommand'] = command
        fitted_model.prepare(local)
        outputs.append(local)
        assert local['iqtree_model'] == g['iqtree_model']
        assert local['omega'] == .8983089442
        assert local['kappa'] == 1.2122738383
        assert local['fitted_model_provenance']['precision'] == 'checkpoint_and_precise_frequencies'
    for local in outputs[1:]:
        np.testing.assert_array_equal(outputs[0]['instantaneous_codon_rate_matrix'], local['instantaneous_codon_rate_matrix'])
        assert outputs[0]['fitted_model_provenance'] == local['fitted_model_provenance']


def test_report_only_is_shared_and_strict_bootstrap_still_requires_precision(tmp_path):
    g = context(tmp_path)
    (tmp_path/'fit.ckp.gz').unlink()
    q = g['instantaneous_codon_rate_matrix'].copy()
    for command in ['search','sites','scan']:
        local = dict(g, subcommand=command)
        fitted_model.prepare(local)
        np.testing.assert_array_equal(local['instantaneous_codon_rate_matrix'], q)
        assert not local.get('iqtree_redo', False)
        assert local['fitted_model_provenance']['precision'] == 'reported_parameters'
    with pytest.raises(ValueError, match='matching .ckp.gz'):
        fitted_model.prepare(g, strict=True)


@pytest.mark.parametrize('key,value', [('omega',2.), ('kappa',3.), ('equilibrium_frequency',np.array([.1,.9]))])
def test_mismatched_fitted_sidecars_are_rejected(tmp_path, key, value):
    g = context(tmp_path)
    g[key] = value
    with pytest.raises(ValueError, match='mismatch'):
        fitted_model.prepare(g)


@pytest.mark.parametrize('model', ['ECMK07_GY0K+F+R4', 'ECMK07_GY0K+FQ'])
def test_ecm_hybrid_scales_all_nonsynonymous_exchangeabilities(tmp_path, model):
    g = context(tmp_path, 'FQ' if model.endswith('FQ') else 'F')
    g.update(
        substitution_model=model, kappa=1.0, omega=0.4,
        codon_orders=np.array(['AAG', 'AAA', 'TTC', 'TTT']),
        synonymous_indices={'K': [0, 1], 'F': [2, 3]},
        equilibrium_frequency=np.array([.3, .1, .2, .4]) if '+F+' in model else np.full(4, .25),
        exchangeability_matrix=np.array([[0, 4, 1, 2], [4, 0, 3, 1], [1, 3, 0, 8], [2, 1, 8, 0.]]),
    )
    with gzip.open(tmp_path/'fit.ckp.gz', 'wt') as handle:
        handle.write('ModelCodon:\n kappa: 1\n omega: 0.4000000123\nOther:\n value: 1\n')
    (tmp_path/'fit.log').write_text('Empirical nonsynonymous/synonymous ratio (omega_E): 0.0613\n'
                                   'Empirical state frequencies: 0.1 0.3 0.2 0.4\n')

    endpoint_io.validate_codon_model(model)
    g.pop('instantaneous_codon_rate_matrix')
    g['omega'] = None  # IQ-TREE does not print the actual multiplier in its report.
    fitted_model.prepare(g)
    q, pi = g['instantaneous_codon_rate_matrix'], g['equilibrium_frequency']
    assert g['omega'] == 0.4000000123
    assert g['fitted_model_provenance']['precision'] == 'checkpoint_and_precise_frequencies'
    exchange = q / pi[None, :]
    # These N transitions differ by multiple nucleotides. Compare ratios to
    # a synonymous rate so the overall normalization cancels.
    np.testing.assert_allclose(exchange[0, 2:] / exchange[0, 1], g['omega'] * np.array([1., 2.]) / 4)
    assert exchange[2, 3] / exchange[0, 1] == pytest.approx(2)
    np.testing.assert_allclose(q.sum(axis=1), 0, atol=1e-15)
    assert pi @ -q.diagonal() == pytest.approx(1)
    assert g['exchangeability_matrix'][0, 2] == 1
    report = tmp_path/'fit.iqtree'
    g['path_iqtree_iqtree'] = str(report)
    report.write_text('Empirical nonsynonymous/synonymous ratio (omega_E): 0.5000\n')
    with pytest.raises(ValueError, match='omega_E mismatch'):
        fitted_model.prepare(g)
    (tmp_path/'fit.ckp.gz').unlink()
    with pytest.raises(ValueError, match='matching IQ-TREE .ckp.gz'):
        fitted_model.prepare(g)


def test_hybrid_checkpoint_frequencies_validate_the_reported_floor(tmp_path):
    g = context(tmp_path)
    g.update(
        substitution_model='ECMK07_GY0K+F+R4', kappa=1.0, omega=None,
        codon_orders=np.array(['AAG', 'AAA', 'TTC', 'TTT']),
        synonymous_indices={'K': [0, 1], 'F': [2, 3]},
        equilibrium_frequency=np.array([.3, .6, 0, .1]),
        exchangeability_matrix=np.array([[0, 4, 1, 2], [4, 0, 3, 1], [1, 3, 0, 8], [2, 1, 8, 0.]]),
    )
    with gzip.open(tmp_path/'fit.ckp.gz', 'wt') as handle:
        handle.write('ModelCodon:\n kappa: 1\n omega: 0.4\n state_freq: 0.5999, 0.3, 0.0001, 0.1\n')
    log = tmp_path/'fit.log'
    raw = 'Empirical state frequencies: 0.6 0.3 0 0.1\n'
    log.write_text(raw)
    with pytest.raises(ValueError, match='frequency mismatch'):
        fitted_model.prepare(g)
    log.write_text('Command: iqtree --inc-zero-freq --min-freq 0.0001\n')
    fitted_model.prepare(g)
    np.testing.assert_allclose(g['equilibrium_frequency'], [.3, .5999, .0001, .1])
    assert g['fitted_model_provenance']['frequency_source'] == 'checkpoint_state_freq'


@pytest.mark.parametrize('model', ['ECMK07_GY+F', 'ECMK07_GY0K+F3X4', 'ECMK07_GY0K'])
def test_unverified_empirical_mechanistic_model_options_are_rejected(model):
    with pytest.raises(ValueError, match='Supported codon models'):
        endpoint_io.validate_codon_model(model)


@pytest.mark.parametrize('redo', [False, True])
def test_explicit_fit_is_authoritative_unless_refit_requested(monkeypatch, redo):
    from csubst import parser_misc
    g = dict(infile_type='iqtree', iqtree_redo=redo, iqtree_model='ECMK07+F+R4')
    g.update({'iqtree_'+suffix: 'fit.'+suffix for suffix in ['iqtree','log','rate','state','treefile']})
    monkeypatch.setattr(parser_misc.parser_iqtree, 'check_intermediate_files', lambda g: (g, True))
    monkeypatch.setattr(parser_misc.parser_iqtree, 'read_iqtree', lambda g, eq: dict(g, substitution_model='GY+F'))
    calls = []
    monkeypatch.setattr(parser_misc.parser_iqtree, 'check_iqtree_dependency', lambda g: None)
    monkeypatch.setattr(parser_misc.parser_iqtree, 'run_iqtree_ancestral', lambda g, **kwargs: calls.append(g))
    parser_misc.generate_intermediate_files(g)
    assert bool(calls) == redo
