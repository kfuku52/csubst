"""Simulation presets and their command-line defaults."""

LEGACY_DEFAULTS = {
    'background_omega': None,
    'foreground_omega': 0.2,
    'num_simulated_site': -1,
    'percent_convergent_site': 100.,
    'tree_scaling_factor': 1.,
    'foreground_scaling_factor': 1.,
    'convergent_amino_acids': 'random1',
    'simulate_asrv': 'no',
    'export_expected_omega': False,
}

_BASE = dict(background_omega=0.2, num_simulated_site=520, percent_convergent_site=0.,
             convergent_amino_acids='random0', export_expected_omega=True)
_BIAS = dict(_BASE, percent_convergent_site=20., convergent_amino_acids='random1')
_ECMK07 = dict(_BASE, background_omega='model', foreground_omega='background')
_EPISTASIS = dict(_ECMK07, num_simulated_site=1200, percent_convergent_site=50.,
                  simulate_focal_tips=24, simulate_focal_branch_length=0.08,
                  simulate_epistasis_distance_scale=0.18, simulate_epistasis_preference_multiplier=8.)

PRESETS = {
    'custom': {},
    'baseline': _BASE,
    'biased': _BIAS,
    'short_alignment': dict(_BASE, num_simulated_site=120),
    'short_alignment_bias': dict(_BIAS, num_simulated_site=120),
    'long_alignment': dict(_BASE, num_simulated_site=1400),
    'short_branches': dict(_BASE, tree_scaling_factor=0.3),
    'short_branches_bias': dict(_BIAS, tree_scaling_factor=0.3),
    'long_branches': dict(_BASE, tree_scaling_factor=2.5),
    'long_branches_bias': dict(_BIAS, tree_scaling_factor=2.5),
    'rate_mixture': dict(_BASE, simulate_asrv='mixture'),
    'rate_mixture_bias': dict(_BIAS, simulate_asrv='mixture'),
    'strong_constraint': dict(_BASE, background_omega=0.05),
    'weak_constraint': dict(_BASE, background_omega=0.6),
    'foreground_acceleration': dict(_BASE, percent_convergent_site=20., foreground_scaling_factor=3.),
    'foreground_relaxation': dict(_BASE, percent_convergent_site=20., foreground_omega=0.8),
    'ecmk07': _ECMK07,
    'ecmk07_rate_mixture': dict(_ECMK07, simulate_asrv='mixture'),
    'ecmk07_biased': dict(_ECMK07, percent_convergent_site=20., convergent_amino_acids='random1'),
    'epistasis_neutral': dict(_EPISTASIS, percent_convergent_site=0.),
    'epistasis_shared': _EPISTASIS,
    'epistasis_inverse': _EPISTASIS,
}


def apply_defaults(g):
    """Apply a named preset with explicit values taking precedence."""
    mode = str(g.get('simulate_scenario', 'custom')).strip().lower()
    if mode not in PRESETS:
        raise ValueError('Unknown simulation scenario: {}.'.format(mode))
    g['simulate_scenario'] = mode
    for key, value in dict(LEGACY_DEFAULTS, **PRESETS[mode]).items():
        if g.get(key) is None:
            g[key] = value


def epistasis_mode(g):
    scenario = g.get('simulate_scenario', 'custom')
    return scenario.removeprefix('epistasis_') if scenario.startswith('epistasis_') else None
