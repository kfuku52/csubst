import copy
import gzip

import numpy as np
import pytest

from csubst import fitted_model


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
