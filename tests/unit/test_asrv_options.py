import pytest

from factories import make_args
from csubst import cli, param


@pytest.mark.parametrize('extra, expected', [
    ([], 'empirical_all'),
    (['--asrv', 'empirical'], 'empirical'),
    (['--asrv', 'empirical_all'], 'empirical_all'),
    (['--output_stat', 'any2spe'], 'empirical_all'),
    (['--asrv', 'each'], 'each'),
    (['--max_arity', '3'], 'empirical_all'),
    (['--max_arity', '3', '--output_stat', 'spe2spe'], 'each'),
    (['--urn_model', 'fisher'], 'each'),
    (['--calc_omega_pvalue', 'yes', '--output_stat', 'any2spe'], 'each'),
])
def test_resolve_search_urn_default(extra, expected):
    args = cli._build_parser().parse_args(['search', '--expectation_method', 'urn', *extra])
    assert param.get_global_parameters(args)['asrv'] == expected


def test_cross_fitted_empirical_still_requires_pairs():
    args = cli._build_parser().parse_args([
        'search', '--expectation_method', 'urn', '--asrv', 'empirical', '--max_arity', '3',
    ])
    with pytest.raises(ValueError, match='branch pairs'):
        param.get_global_parameters(args)


@pytest.mark.parametrize('mode', ['empirical', 'empirical_all'])
def test_explicit_empirical_rejects_incompatible_statistics(mode):
    args = cli._build_parser().parse_args([
        'search', '--expectation_method', 'urn', '--asrv', mode, '--output_stat', 'spe2spe',
    ])
    with pytest.raises(ValueError, match='destination|any2any/any2spe/any2dif'):
        param.get_global_parameters(args)


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


def test_cli_accepts_quadrature_and_shared_rate_draws():
    args = cli._build_parser(show_advanced=True).parse_args([
        'search', '--expectation_method', 'urn', '--asrv', 'file',
        '--urn_wallenius_expectation', 'quadrature', '--asrv_posterior_draws', '64',
    ])
    result = param.get_global_parameters(args)
    assert result['urn_wallenius_expectation'] == 'quadrature'
    assert result['asrv_posterior_draws'] == 64


@pytest.mark.parametrize('additional, expected', [
    (['--substitution_posterior', 'marginal'], 'requires joint'),
    (['--calibrate_longtail', 'yes'], 'external pipeline calibration'),
    (['--asrv_report', 'yes'], 'fixed-weight'),
])
def test_shared_rate_draws_reject_incompatible_estimators(additional, expected):
    args = cli._build_parser(show_advanced=True).parse_args([
        'search', '--expectation_method', 'urn', '--asrv', 'file',
        '--asrv_posterior_draws', '64', *additional,
    ])
    with pytest.raises(ValueError, match=expected):
        param.get_global_parameters(args)


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
    with pytest.raises(ValueError, match='structure cross-fitting defines its own clade training sets'):
        param.get_global_parameters(make_args(expectation_method='urn', asrv_training_branches='background', epistasis_beta='1'))
