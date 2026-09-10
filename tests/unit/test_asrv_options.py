import pytest

from factories import make_args
from csubst import cli, param


def test_cli_and_parameters_accept_training_concentration_report():
    args = cli._build_parser(show_advanced=True).parse_args([
        'search', '--expectation_method', 'urn', '--asrv_training_branches', 'background',
        '--asrv_concentration', '2', '--asrv_report', 'yes',
        '--urn_wallenius_expectation', 'exact',
    ])
    result = param.get_global_parameters(args)
    assert result['asrv_concentration'] == 2
    assert result['asrv_report'] is True
    assert result['asrv_training_branches'] == 'background'
    assert result['urn_wallenius_expectation'] == 'exact'


@pytest.mark.parametrize('options,match', [
    ({'asrv_concentration': -1}, 'concentration'),
    ({'asrv_concentration': float('nan')}, 'concentration'),
    ({'asrv_concentration': float('inf')}, 'concentration'),
    ({'asrv': 'pool', 'asrv_concentration': 2}, 'only to'),
    ({'asrv': 'file', 'asrv_training_branches': 'background'}, 'empirical'),
    ({'asrv_training_branches': '1,1'}, 'duplicate'),
    ({'urn_model': 'fisher', 'urn_wallenius_expectation': 'exact'}, 'Wallenius'),
])
def test_invalid_asrv_combinations_fail(options, match):
    with pytest.raises(ValueError, match=match):
        param.get_global_parameters(make_args(expectation_method='urn', **options))


def test_custom_asrv_on_codon_model_fails_instead_of_being_ignored():
    with pytest.raises(ValueError, match='require --expectation_method urn'):
        param.get_global_parameters(make_args(asrv_concentration=2))


def test_external_recoding_training_requires_auto_scheme():
    with pytest.raises(ValueError, match='requires srchisq6 or kgbauto6'):
        param.get_global_parameters(make_args(nonsyn_recode_training_alignment='train.fa'))
    result = param.get_global_parameters(make_args(nonsyn_recode='srchisq6', nonsyn_recode_training_alignment='train.fa'))
    assert result['nonsyn_recode_training_alignment'] == 'train.fa'


def test_epistasis_cannot_silently_reintroduce_foreground_training():
    with pytest.raises(ValueError, match='epistasis fitting still uses all branches'):
        param.get_global_parameters(make_args(expectation_method='urn', asrv_training_branches='background', epistasis_beta='1'))
