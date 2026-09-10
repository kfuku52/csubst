"""Selection diagnostics against materialized sitewise count contributions."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from csubst import ete, omega, parser_misc, sequence, site_filter, substitution, tree


def _fixture(tmp_path, native=False):
    tr = tree.add_numerical_node_labels(ete.PhyloNode('((A:.5,B:.5)X:.5,(C:.5,D:.5)Y:.5)R;', format=1))
    ids = {node.name: int(ete.get_prop(node, 'numerical_label')) for node in tr.traverse()}
    state = np.zeros((7, 3, 4))
    state[:, :, 0] = 1
    for name in ('X', 'Y'):
        state[ids[name], 0] = [.8, 0, .2, 0]
    for node in ete.iter_leaves(tr):
        last = 1 if node.name in ('A', 'C') else 2
        state[ids[node.name], 2] = np.eye(4)[last]
        ete.set_prop(node, 'sequence', 'AAAAAA' + ('AAG' if last == 1 else 'AAC'))
    g = dict(tree=tr, num_input_site=3, float_type=np.float64, float_tol=1e-12,
             state_cdn=state, codon_orders=np.array(['AAA', 'AAG', 'AAC', 'AAT']),
             amino_acid_orders=['K', 'N'], synonymous_indices={'K': [0, 1], 'N': [2, 3]},
             nonsynonymous_indices={'K': [0, 1], 'N': [2, 3]}, nonsyn_state_orders=np.array(['K', 'N']),
             max_synonymous_size=2, threads=1, min_sub_pp=0, nonsyn_recode='no',
             iqtree_rate_values=np.ones(3), expectation_method='codon_model',
             instantaneous_codon_rate_matrix=(np.ones((4, 4)) - 4 * np.eye(4)) / 3,
             instantaneous_nsy_rate_matrix=np.ones((2, 2)) - 2 * np.eye(2),
             site_filter_report=True, drop_invariant_tip_sites=False,
             output_stats=['any2any', 'spe2any', 'any2spe', 'spe2spe', 'dif2dif'],
             outdir=str(tmp_path), output_prefix='audit', current_arity=2, max_arity=2,
             expected_state_backend='expm')
    g['state_pep'] = sequence.cdn2pep_state(state, g)
    g['state_nsy'] = g['state_pep'].copy()
    if native:
        g.update(nonsyn_recode='3di20', sa_asr_mode='direct', sa_iqtree_model='GTR',
                 **{'3di_q': (np.ones((20, 20)) - 20 * np.eye(20)) / 19,
                    '3di_pi': np.ones(20) / 20, '3di_orders': np.array(list('ACDEFGHIKLMNPQRSTVWY')),
                    '3di_rates': np.ones(3), '3di_branch_lengths': np.full(7, .5),
                    '3di_tip_invariant_mask': np.array([True, True, True])})
        g['3di_branch_lengths'][ids['R']] = 0
        nsy = np.zeros((7, 3, 20))
        nsy[:, :, 0] = 1
        for name in ('X', 'Y'):
            nsy[ids[name], 0, :2] = [.8, .2]
        g['state_nsy'] = nsy
    return g, np.array([[ids['A'], ids['C']]], dtype=np.int64)


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('backend', ['expm', 'eigen'])
def test_fixed_model_report_matches_sitewise_counts_and_preserves_analysis(tmp_path, native, backend):
    g, ids = _fixture(tmp_path, native=native)
    g['expected_state_backend'] = backend
    state_before = g['state_cdn'].copy()
    site_filter.prepare(g)
    np.testing.assert_array_equal(g['state_cdn'], state_before)
    on = substitution.get_substitution_tensor(g['state_nsy'], mode='asis', g=g)
    os = substitution.get_substitution_tensor(g['state_cdn'], mode='syn', g=g)
    tree.rescale_branch_length(g, os, on)
    before = copy.deepcopy(g)
    cb = pd.DataFrame(ids, columns=['branch_id_1', 'branch_id_2'])
    actual = omega.get_E(cb.copy(), g, on, os)
    before['site_filter_report'] = False
    baseline = omega.get_E(cb.copy(), before, on, os)
    pd.testing.assert_frame_equal(actual, baseline)
    report = pd.read_csv(tmp_path / 'audit_site_filter_counts_2.tsv', sep='\t')
    for criterion in site_filter.CRITERIA:
        rows = report.loc[report.criterion == criterion].set_index('partition')
        counts = [col for col in rows if col.startswith(('OCN', 'OCS', 'ECN', 'ECS'))]
        np.testing.assert_allclose(rows.loc['all', counts].astype(float),
                                   rows.loc['retained', counts].astype(float) +
                                   rows.loc['excluded', counts].astype(float), atol=1e-12)
        for col in actual:
            if col.startswith('EC'):
                assert rows.loc['all', col] == pytest.approx(actual.loc[0, col], abs=1e-12)
    # Independent materialized tensor path, with masks applied after projection.
    for channel, mode, submode in [('N', 'nsy', 'asis'), ('S', 'cdn', 'syn')]:
        expected_state = omega.get_exp_state(before, mode)
        tensor = substitution.get_substitution_tensor(expected_state,
                  state_tensor_anc=before['state_nsy' if channel == 'N' else 'state_cdn'], mode=submode, g=before)
        site_counts = substitution._get_sparse_site_vectors(tensor, ids[0])[0]
        for criterion, mask in g['_site_filter_masks'].items():
            excluded = report.loc[(report.criterion == criterion) & (report.partition == 'excluded')]
            assert excluded['EC' + channel + 'any2any'].iloc[0] == pytest.approx(site_counts[mask].sum())
    dropped = report.loc[(report.criterion == 'zero_sub_mass') & (report.partition == 'excluded')].iloc[0]
    assert dropped.OCNany2any == 0
    assert dropped.OCSany2any == 0
    assert dropped.ECNany2any > 0
    assert dropped.ECSany2any > 0
    tip = report.loc[(report.criterion == 'tip_invariant') & (report.partition == 'excluded')].iloc[0]
    assert tip.OCNany2any > 0
    if native:
        # A structurally invariant column can carry synonymous signal.
        assert tip.OCSany2any > 0
        assert tip.num_sites == 3  # Empty retained partition is a valid diagnostic.
    metadata = json.loads((tmp_path / 'audit_site_filter.json').read_text())
    assert metadata['selection_repeated_in_null'] is False
    assert metadata['analysis_site_selection'] == 'no'


def test_zero_mass_filter_changes_remaining_site_expectation(tmp_path):
    g, ids = _fixture(tmp_path)
    def expectations(local):
        on = substitution.get_substitution_tensor(local['state_nsy'], mode='asis', g=local)
        os = substitution.get_substitution_tensor(local['state_cdn'], mode='syn', g=local)
        tree.rescale_branch_length(local, os, on)
        se = omega.get_exp_state(local, 'cdn')
        te = substitution.get_substitution_tensor(se, state_tensor_anc=local['state_cdn'], mode='syn', g=local)
        return substitution._get_sparse_site_vectors(te, ids[0])[0]
    full = expectations(copy.deepcopy(g))
    filtered = copy.deepcopy(g)
    filtered['drop_invariant_tip_sites_mode'] = 'zero_sub_mass'
    parser_misc.drop_invariant_tip_sites(filtered)
    retained = expectations(filtered)
    assert retained[-1] > full[-1]


def test_report_uses_requested_base_stats_and_no_excluded_sites(tmp_path):
    g, ids = _fixture(tmp_path)
    g['output_stats'] = ['any2any']
    g['float_tol'] = 0
    for node in ete.iter_leaves(g['tree']):
        ete.set_prop(node, 'sequence', 'AAG' * 3 if node.name == 'A' else 'AAC' * 3)
    site_filter.prepare(g)
    g['_site_filter_masks']['zero_sub_mass'][:] = False
    on = substitution.get_substitution_tensor(g['state_nsy'], mode='asis', g=g)
    os = substitution.get_substitution_tensor(g['state_cdn'], mode='syn', g=g)
    tree.rescale_branch_length(g, os, on)
    omega.get_E(pd.DataFrame(ids, columns=['branch_id_1', 'branch_id_2']), g, on, os)
    report = pd.read_csv(tmp_path / 'audit_site_filter_counts_2.tsv', sep='\t')
    assert (report.loc[report.partition == 'excluded', 'ECNany2any'] == 0).all()
    assert 'ECNspe2spe' not in report
