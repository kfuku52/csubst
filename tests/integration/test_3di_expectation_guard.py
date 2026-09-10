"""3Di state symbols must never be evaluated with a codon-derived N model."""

import numpy as np
import pandas as pd
import pytest

from factories import make_args
from csubst import main_analyze, omega, param, parser_misc, substitution_scan


@pytest.mark.parametrize('command', ['search', 'analyze'])
@pytest.mark.parametrize('asr_mode', ['translate'])
@pytest.mark.parametrize('recode', ['3di20', '3di', 'structural_alphabet'])
def test_search_rejects_3di_codon_expectations_before_loading_input(command, asr_mode, recode):
    with pytest.raises(ValueError, match='require --sa_asr_mode direct'):
        param.get_global_parameters(make_args(
            subcommand=command, nonsyn_recode=recode, sa_asr_mode=asr_mode,
            full_cds_alignment_file='missing.fa',
        ))


@pytest.mark.parametrize('command', ['search', 'analyze', 'scan', 'inspect', 'sites'])
def test_3di_urn_and_commands_without_omega_expectations_remain_available(command):
    method = 'urn' if command in ['search', 'analyze'] else 'codon_model'
    g = param.get_global_parameters(make_args(
        subcommand=command, nonsyn_recode='3di20', expectation_method=method,
        full_cds_alignment_file='missing.fa',
    ))
    assert g['expectation_method'] == method
    assert g['nonsyn_recode'] == '3di20'


def test_direct_search_entry_rejects_3di_before_creating_output(tmp_path):
    outdir = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='require --sa_asr_mode direct'):
        main_analyze.main_analyze({'nonsyn_recode': '3di20', 'sa_asr_mode': 'translate', 'outdir': str(outdir)})
    assert not outdir.exists()


def test_get_E_rejects_cached_codon_reducers_for_3di():
    g = {'nonsyn_recode': '3di20', 'expectation_method': 'codon_model',
         'EN_reducer': {}, 'ES_reducer': {}}
    with pytest.raises(ValueError, match='Missing fitted 3Di'):
        omega.get_E(pd.DataFrame(), g, None, None)


@pytest.mark.parametrize('backend', ['dense', 'sparse', 'reducer'])
def test_expected_state_backends_cannot_bypass_state_space_check(backend):
    # Even an injected/stale rate matrix cannot turn 3Di into a codon state space.
    g = {'nonsyn_recode': '3di20', 'instantaneous_nsy_rate_matrix': np.eye(20)}
    with pytest.raises(ValueError, match='Missing fitted 3Di'):
        if backend == 'dense':
            omega.get_exp_state(g, mode='nsy')
        elif backend == 'sparse':
            omega._get_fused_expected_sparse_substitution_tensor(g, mode='nsy')
        else:
            omega._get_fused_expected_sparse_reducer(g, mode='nsy', selected_base_stats=['any2spe'])


def test_codon_matrix_cannot_be_converted_to_3di():
    with pytest.raises(ValueError, match='does not define transition rates for 3Di'):
        parser_misc.cdn2nsy_matrix(np.eye(20), {'nonsyn_recode': '3di20'})


def test_scan_retains_codon_context_without_creating_a_3di_Q(monkeypatch):
    def load_metadata(g):
        g.update({
            'substitution_model': 'GY', 'omega': 1., 'kappa': 1.,
            'equilibrium_frequency': np.array([.5, .5]),
            'codon_orders': np.array(['AAA', 'AAC']),
            'amino_acid_orders': np.array(['K', 'N']),
            'codon_table': [('K', 'AAA'), ('N', 'AAC')],
            'reconstruction_codon_table': [('K', 'AAA'), ('N', 'AAC')],
            'synonymous_indices': {'K': [0], 'N': [1]},
            'matrix_groups': {'K': ['AAA'], 'N': ['AAC']},
            'max_synonymous_size': 1,
        })
        return g

    monkeypatch.setattr(parser_misc.parser_iqtree, 'get_input_information', load_metadata)
    g = parser_misc.read_input({
        'infile_type': 'iqtree', 'expectation_method': 'codon_model',
        'subcommand': 'scan', 'nonsyn_recode': '3di20',
        'float_tol': 1e-12, 'float_type': np.float64,
        'instantaneous_nsy_rate_matrix': np.eye(20), 'rate_nsy_tensor': np.ones((1, 20, 20)),
    })
    assert 'instantaneous_nsy_rate_matrix' not in g
    assert 'rate_nsy_tensor' not in g
    np.testing.assert_allclose(g['instantaneous_codon_rate_matrix'].sum(axis=1), 0)
    np.testing.assert_allclose(g['instantaneous_aa_rate_matrix'], g['instantaneous_codon_rate_matrix'])
    assert substitution_scan.resolve_scan_rate_exposure(g) == 'state_aware'
    assert substitution_scan._build_scan_q_context(g, 'state_aware')['q_matrix'] is None


@pytest.mark.parametrize('recode', ['3di20', '3di', 'structural_alphabet'])
def test_default_model_method_accepts_direct_3di(recode):
    g = param.get_global_parameters(make_args(
        subcommand='search', nonsyn_recode=recode, sa_asr_mode='direct',
        full_cds_alignment_file='missing.fa',
    ))
    assert g['expectation_method'] == 'codon_model'
    assert g['nonsyn_recode'] == '3di20'


@pytest.mark.parametrize('model', ['GTR+G4', 'GTR+R4', 'GTR+FO', 'LG'])
def test_unvalidated_3di_models_fail_before_inference(model):
    with pytest.raises(ValueError, match='currently require uniform GTR'):
        param.get_global_parameters(make_args(
            subcommand='search', nonsyn_recode='3di20', sa_iqtree_model=model,
            full_cds_alignment_file='missing.fa',
        ))
