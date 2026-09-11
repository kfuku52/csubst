import pytest
from csubst import cli, endpoint_io, param, substitution_scan


@pytest.mark.parametrize('command', ['search', 'analyze', 'sites', 'site', 'inspect', 'benchmark', 'scan'])
@pytest.mark.parametrize('posterior', ['joint', 'marginal'])
def test_common_posterior_default_and_opt_out(command, posterior):
    flags = [] if posterior == 'joint' else ['--substitution_posterior', posterior]
    args = cli._build_parser().parse_args([command, *flags])
    assert args.substitution_posterior == posterior
    g = param._normalize_state_parameters(vars(args).copy())
    if command == 'scan':
        assert g['scan_observation'] == posterior
        assert g['scan_rate_exposure'] == ('endpoint' if posterior == 'joint' else 'q_weighted')
        assert g['scan_rate_length'] == ('raw' if posterior == 'joint' else 'n_rescaled')
        substitution_scan.validate_scan_configuration(g)
        assert not endpoint_io.enabled(g)
    else:
        assert endpoint_io.enabled(g) == (posterior == 'joint')


@pytest.mark.parametrize('observation', ['marginal', 'joint', 'bridge'])
def test_scan_override_is_preserved(observation):
    args = cli._build_parser().parse_args(['scan', '--scan_observation', observation])
    g = param._normalize_state_parameters(vars(args).copy())
    assert g['scan_observation'] == observation
    substitution_scan.validate_scan_configuration(g)


def test_explicit_incompatible_scan_exposure_is_not_replaced():
    args = cli._build_parser().parse_args(['scan', '--scan_rate_exposure', 'q_weighted'])
    g = param._normalize_state_parameters(vars(args).copy())
    with pytest.raises(ValueError, match='endpoint'):
        substitution_scan.validate_scan_configuration(g)
