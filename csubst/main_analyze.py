import numpy as np
import pandas as pd

import os
import shutil
import sys
import time

from csubst import combination
from csubst import foreground
from csubst import genetic_code
from csubst import omega
from csubst import param
from csubst import parser_misc
from csubst import sequence
from csubst import substitution
from csubst import table
from csubst import ete
from csubst import output_stat
from csubst import tree


def _remap_site_column_to_alignment(df, g, column_name='site'):
    if column_name not in df.columns:
        return df
    if df.shape[0] == 0:
        return df
    out = df.copy(deep=True)
    site_values = out.loc[:, column_name].to_numpy(dtype=np.int64, copy=True)
    mapped_values = parser_misc.map_internal_site_indices(
        site_indices=site_values,
        g=g,
        missing_value=-1,
        allow_invalid=False,
    )
    out.loc[:, column_name] = mapped_values
    return out


def _plot_state_tree_in_directory(output_dir, state, orders, mode, g):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    cwd = os.getcwd()
    try:
        os.chdir(output_dir)
        tree.plot_state_tree(state=state, orders=orders, mode=mode, g=g)
    finally:
        os.chdir(cwd)


def cb_search(g, b, OS_tensor, ON_tensor, id_combinations, write_cb=True):
    if int(g['max_arity']) < 2:
        raise ValueError('--max_arity should be >= 2.')
    if int(g['max_combination']) < 1:
        raise ValueError('--max_combination should be >= 1.')
    OS_tensor_reducer = substitution.get_reducer_sub_tensor(sub_tensor=OS_tensor, g=g, label='OS')
    ON_tensor_reducer = substitution.get_reducer_sub_tensor(sub_tensor=ON_tensor, g=g, label='ON')
    for current_arity in np.arange(2, g['max_arity'] + 1):
        start = time.time()
        g['current_arity'] = current_arity
        g = param.initialize_df_cb_stats(g)
        print("Arity (K) = {:,}: Generating cb table".format(current_arity), flush=True)
        if (current_arity==2):
            if (g['exhaustive_until']<current_arity)&(g['foreground'] is not None):
                txt = 'Arity (K) = {:,}: Targeted search of foreground branch combinations'
                print(txt.format(current_arity), flush=True)
                g['df_cb_stats'].at[0, 'mode'] = 'foreground'
                g,id_combinations = combination.get_node_combinations(g=g, target_id_dict=g['target_ids'],
                                                                      arity=current_arity, check_attr='name')
            else:
                txt = 'Arity (K) = {:,}: Exhaustive search with all independent branch combinations'
                print(txt.format(current_arity), flush=True)
                g['df_cb_stats'].at[0, 'mode'] = 'exhaustive'
                g,id_combinations = combination.get_node_combinations(g=g, exhaustive=True,
                                                                      arity=current_arity, check_attr="name")
        elif (current_arity >= 3):
            id_columns = cb.columns[cb.columns.str.startswith('branch_id_')].tolist()
            fg_columns = cb.columns[cb.columns.str.startswith('is_fg_')].tolist()
            mf_columns = cb.columns[cb.columns.str.startswith('is_mf_')].tolist()
            mg_columns = cb.columns[cb.columns.str.startswith('is_mg_')].tolist()
            cutoff_stat_entries = table.parse_cutoff_stat(cutoff_stat_str=g['cutoff_stat'])
            cutoff_stat_exp = [item[0] for item in cutoff_stat_entries]
            stat_columns = cb.columns[cb.columns.str.fullmatch('|'.join(cutoff_stat_exp), na=False)].tolist()
            cb_passed_columns = id_columns + fg_columns + mf_columns + mg_columns + stat_columns
            if (g['exhaustive_until'] < current_arity):
                is_stat_enough = table.get_cutoff_stat_bool_array(cb=cb, cutoff_stat_str=g['cutoff_stat'])
                num_branch_ids = is_stat_enough.sum()
                txt = 'Arity (K) = {:,}: Heuristic search with {:,} K-1 branch combinations that passed cutoff stats ({})'
                print(txt.format(current_arity, num_branch_ids, g['cutoff_stat']), flush=True)
                g['df_cb_stats'].at[0, 'mode'] = 'branch_and_bound'
                if is_stat_enough.sum() > g['max_combination']:
                    txt = 'Arity (K) = {:,}: Search will be limited to {:,} of {:,} K-1 branch combinations (see --max_combination)\n'
                    txt = txt.format(current_arity, g['max_combination'], is_stat_enough.sum())
                    sys.stderr.write(txt)
                    cb_passed = cb.loc[is_stat_enough, :].sort_values(by=stat_columns, ascending=False).reset_index(drop=True)
                    cb_passed = cb_passed.iloc[:g['max_combination'], :].loc[:, cb_passed_columns].reset_index(drop=True)
                else:
                    cb_passed = cb.loc[is_stat_enough,cb_passed_columns].reset_index(drop=True)
                if len(set(cb_passed.loc[:,id_columns].values.ravel().tolist())) < current_arity:
                    cb = pd.DataFrame()
                    txt = 'Arity (K) = {:,}: No branch combination satisfied --cutoff_stat. Ending higher-order search at K = {:,}.'
                    print(txt.format(current_arity, current_arity))
                    break
                g,id_combinations = combination.get_node_combinations(g=g, cb_passed=cb_passed, cb_all=False,
                                                                      arity=current_arity, check_attr='name')
            else:
                txt = 'Arity (K) = {:,}: Exhaustive search with {:,} K-1 branch combinations'
                print(txt.format(current_arity, cb.shape[0]))
                g['df_cb_stats'].at[0, 'mode'] = 'exhaustive'
                cb_passed = cb.loc[:,cb_passed_columns].reset_index(drop=True)
                g,id_combinations = combination.get_node_combinations(g=g, cb_passed=cb_passed, cb_all=True,
                                                                      arity=current_arity, check_attr='name')
        else:
            raise ValueError('Invalid arity: {}'.format(current_arity))
        if id_combinations.shape[0] == 0:
            cb = pd.DataFrame()
            txt = 'Arity (K) = {:,}: No branch combination satisfied phylogenetic independence. Ending higher-order search at K = {:,}.'
            print(txt.format(current_arity, current_arity))
            break
        print('Preparing OCS table with up to {:,} process(es).'.format(g['threads']), flush=True)
        cbOS = substitution.get_cb(
            id_combinations,
            OS_tensor_reducer,
            g,
            'OCS',
            selected_base_stats=g.get('output_base_stats'),
        )
        print('Preparing OCN table with up to {:,} process(es).'.format(g['threads']), flush=True)
        cbON = substitution.get_cb(
            id_combinations,
            ON_tensor_reducer,
            g,
            'OCN',
            selected_base_stats=g.get('output_base_stats'),
        )
        cb = table.merge_tables(cbOS, cbON)
        del cbOS, cbON
        cb = substitution.add_dif_stats(cb, g['float_tol'], prefix='OC', output_stats=g.get('output_stats'))
        cb, g = omega.calc_omega(cb, OS_tensor_reducer, ON_tensor_reducer, g)
        if bool(g.get('epistasis_enabled', False)):
            g['df_cb_stats'].at[0, 'epistasis_enabled'] = 'Y'
            g['df_cb_stats'].at[0, 'epistasis_apply_to'] = str(g.get('epistasis_apply_to', 'N'))
            g['df_cb_stats'].at[0, 'epistasis_site_metric'] = str(g.get('epistasis_site_metric_resolved', g.get('epistasis_site_metric', 'auto')))
            for channel in ['N', 'S']:
                channel_state = g.get('_epistasis_state', dict()).get(channel, None)
                if channel_state is None:
                    continue
                g['df_cb_stats'].at[0, 'epistasis_beta_' + channel] = float(channel_state.get('beta', np.nan))
                g['df_cb_stats'].at[0, 'epistasis_clip_' + channel] = float(channel_state.get('clip', np.nan))
        if (g['calibrate_longtail']):
            if (g['exhaustive_until'] >= current_arity):
                cb = omega.calibrate_dsc(cb, output_stats=g.get('output_stats'))
                if bool(g.get('calc_omega_pvalue', False)):
                    # Recompute empirical p/q values on calibrated omegaC columns.
                    # Non-calibrated p/q columns are retained as *_nocalib by calibrate_dsc().
                    cb = omega.add_omega_empirical_pvalues(
                        cb=cb,
                        ON_tensor=ON_tensor_reducer,
                        OS_tensor=OS_tensor_reducer,
                        g=g,
                    )
                g['df_cb_stats'].at[0, 'dSC_calibration'] = 'Y'
            else:
                txt = '--calibrate_longtail is deactivated for arity = {}. '
                txt += 'This option is effective for the arity range specified by --exhaustive_until.\n'
                sys.stderr.write(txt.format(current_arity))
                g['df_cb_stats'].at[0, 'dSC_calibration'] = 'N'
        else:
            g['df_cb_stats'].at[0, 'dSC_calibration'] = 'N'
        if g['branch_dist']:
            cb = tree.get_node_distance(
                tree=g['tree'],
                cb=cb,
                ncpu=g['threads'],
                float_type=g['float_type'],
                min_items_for_parallel=int(g.get('parallel_min_items_branch_dist', 20000)),
                min_items_per_job=int(g.get('parallel_min_items_per_job_branch_dist', 5000)),
            )
        cb = substitution.get_substitutions_per_branch(cb, b, g)
        #cb = combination.calc_substitution_patterns(cb)
        cb = table.get_linear_regression(cb)
        cb = output_stat.drop_unrequested_stat_columns(cb, g.get('output_stats'))
        cb, g = foreground.get_foreground_branch_num(cb, g)
        cb = table.sort_cb(cb)
        if write_cb:
            file_name = "csubst_cb_" + str(current_arity) + ".tsv"
            cb_column_original = cb.columns.tolist()
            cb.columns = cb.columns.str.replace('_PLACEHOLDER', '')
            cb.to_csv(file_name, sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
            cb.columns = cb_column_original
            txt = 'Memory consumption of cb table: {:,.1f} Mbytes (dtype={})'
            print(txt.format(cb.values.nbytes/1024/1024, cb.values.dtype), flush=True)
        g = foreground.add_median_cb_stats(g, cb, current_arity, start)
        if (g['fg_clade_permutation']>0):
            g = foreground.clade_permutation(
                cb=cb,
                g=g,
                OS_tensor_reducer=OS_tensor_reducer,
                ON_tensor_reducer=ON_tensor_reducer,
            )
        g['df_cb_stats'] = g['df_cb_stats'].loc[:, sorted(g['df_cb_stats'].columns.tolist())]
        g['df_cb_stats_main'] = pd.concat([g['df_cb_stats_main'], g['df_cb_stats']], ignore_index=True)
        if current_arity == g['max_arity']:
            txt = 'Maximum arity (K = {:,}) reached. Ending higher-order search of branch combinations.'
            print(txt.format(g['max_arity']))
            break
    return g,cb


def _annotate_branch_length_column(b, tree_obj):
    branch_length_by_id = {}
    for node in tree_obj.traverse():
        node_id = int(ete.get_prop(node, "numerical_label"))
        node_dist = 0.0 if (node.dist is None) else float(node.dist)
        branch_length_by_id[node_id] = node_dist
    b.loc[:, 'branch_length'] = b.loc[:, 'branch_id'].map(branch_length_by_id)
    return b


def _standardize_degree_values(values, float_tol):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    z_values = np.zeros(shape=values.shape, dtype=np.float64)
    is_finite = np.isfinite(values)
    if not is_finite.any():
        return z_values
    finite_values = values[is_finite]
    mean_value = finite_values.mean()
    std_value = finite_values.std(ddof=0)
    if std_value > float(float_tol):
        z_values[is_finite] = (finite_values - mean_value) / std_value
    else:
        z_values[is_finite] = 0.0
    return z_values


def _map_epistasis_profile_to_internal(df, num_site, site_index_alignment, value_col):
    value_series = pd.to_numeric(df.loc[:, value_col], errors='coerce')
    profile_internal = np.full(shape=(num_site,), fill_value=np.nan, dtype=np.float64)
    if ('site' in df.columns) and ('codon_site_alignment' not in df.columns):
        site_values = pd.to_numeric(df.loc[:, 'site'], errors='coerce')
        valid = site_values.notna() & value_series.notna()
        site_idx = site_values.loc[valid].astype(np.int64).to_numpy(copy=False)
        value_array = value_series.loc[valid].to_numpy(dtype=np.float64, copy=False)
        is_valid_site = (site_idx >= 0) & (site_idx < num_site)
        profile_internal[site_idx[is_valid_site]] = value_array[is_valid_site]
        return profile_internal
    if 'codon_site_alignment' not in df.columns:
        raise ValueError('codon_site_alignment column was not found in epistasis degree file.')
    aln_values = pd.to_numeric(df.loc[:, 'codon_site_alignment'], errors='coerce')
    valid = aln_values.notna() & value_series.notna()
    aln_site = aln_values.loc[valid].astype(np.int64).to_numpy(copy=False)
    value_array = value_series.loc[valid].to_numpy(dtype=np.float64, copy=False)
    row_df = pd.DataFrame({
        'codon_site_alignment': aln_site,
        'value': value_array,
    })
    row_df = row_df.loc[row_df['codon_site_alignment'] > 0, :]
    row_df = row_df.groupby('codon_site_alignment', as_index=False)['value'].mean()
    value_by_alignment = {
        int(aln_site): float(value)
        for aln_site, value in zip(row_df['codon_site_alignment'].tolist(), row_df['value'].tolist())
    }
    for site_idx in range(num_site):
        aln_site_1based = int(site_index_alignment[site_idx]) + 1
        if aln_site_1based in value_by_alignment:
            profile_internal[site_idx] = value_by_alignment[aln_site_1based]
    return profile_internal


def _extract_epistasis_profile_from_df(df, num_site, site_index_alignment, z_col_candidates, raw_col_candidates, float_tol):
    z_col = next((col for col in z_col_candidates if col in df.columns), None)
    raw_col = next((col for col in raw_col_candidates if col in df.columns), None)
    if (z_col is None) and (raw_col is None):
        return None
    if z_col is not None:
        profile_internal = _map_epistasis_profile_to_internal(
            df=df,
            num_site=num_site,
            site_index_alignment=site_index_alignment,
            value_col=z_col,
        )
        out = np.zeros(shape=(num_site,), dtype=np.float64)
        finite = np.isfinite(profile_internal)
        out[finite] = profile_internal[finite]
        return out
    raw_internal = _map_epistasis_profile_to_internal(
        df=df,
        num_site=num_site,
        site_index_alignment=site_index_alignment,
        value_col=raw_col,
    )
    return _standardize_degree_values(values=raw_internal, float_tol=float_tol)


def _resolve_epistasis_profile_map_from_df(df, g, num_site):
    site_index_alignment = parser_misc.get_site_index_alignment(g=g, expected_num_site=num_site)
    profile_specs = {
        'degree': {
            'z': ['epistasis_contact_degree_z', 'contact_degree_z', 'epistasis_degree_z'],
            'raw': ['epistasis_contact_degree', 'contact_degree', 'epistasis_degree'],
        },
        'proximity': {
            'z': ['epistasis_contact_proximity_z', 'contact_proximity_z', 'epistasis_proximity_z'],
            'raw': ['epistasis_contact_proximity', 'contact_proximity', 'epistasis_proximity'],
        },
    }
    profile_map = dict()
    for profile_name, spec in profile_specs.items():
        profile = _extract_epistasis_profile_from_df(
            df=df,
            num_site=num_site,
            site_index_alignment=site_index_alignment,
            z_col_candidates=spec['z'],
            raw_col_candidates=spec['raw'],
            float_tol=float(g['float_tol']),
        )
        if profile is not None:
            profile_map[profile_name] = np.asarray(profile, dtype=np.float64).reshape(-1)
    return profile_map


def _select_epistasis_profile(profile_map, metric, float_tol):
    metric = str(metric).strip().lower()
    degree = profile_map.get('degree', None)
    proximity = profile_map.get('proximity', None)
    if metric == 'degree':
        if degree is None:
            raise ValueError('Requested --epistasis_site_metric degree but degree profile was not found.')
        return np.asarray(degree, dtype=np.float64).reshape(-1), 'degree'
    if metric == 'proximity':
        if proximity is None:
            raise ValueError('Requested --epistasis_site_metric proximity but proximity profile was not found.')
        return np.asarray(proximity, dtype=np.float64).reshape(-1), 'proximity'
    if metric == 'hybrid':
        if (degree is None) and (proximity is None):
            raise ValueError('Requested --epistasis_site_metric hybrid but no epistasis profile was found.')
        if degree is None:
            print('Hybrid epistasis profile requested, but degree profile was unavailable. Falling back to proximity.', flush=True)
            return np.asarray(proximity, dtype=np.float64).reshape(-1), 'proximity'
        if proximity is None:
            print('Hybrid epistasis profile requested, but proximity profile was unavailable. Falling back to degree.', flush=True)
            return np.asarray(degree, dtype=np.float64).reshape(-1), 'degree'
        hybrid = 0.5 * (np.asarray(degree, dtype=np.float64) + np.asarray(proximity, dtype=np.float64))
        hybrid = _standardize_degree_values(values=hybrid, float_tol=float_tol)
        return hybrid.astype(np.float64, copy=False), 'hybrid'
    if metric == 'auto':
        if (degree is not None) and (proximity is not None):
            hybrid = 0.5 * (np.asarray(degree, dtype=np.float64) + np.asarray(proximity, dtype=np.float64))
            hybrid = _standardize_degree_values(values=hybrid, float_tol=float_tol)
            return hybrid.astype(np.float64, copy=False), 'hybrid'
        if proximity is not None:
            return np.asarray(proximity, dtype=np.float64).reshape(-1), 'proximity'
        if degree is not None:
            return np.asarray(degree, dtype=np.float64).reshape(-1), 'degree'
        raise ValueError('No epistasis profile was available for --epistasis_site_metric auto.')
    raise ValueError('Unsupported --epistasis_site_metric: {}'.format(metric))


def _load_epistasis_degree_from_file(g, num_site):
    degree_file = str(g.get('epistasis_degree_file', '')).strip()
    if degree_file == '':
        return None
    if not os.path.exists(degree_file):
        raise FileNotFoundError('Epistasis degree file was not found: {}'.format(degree_file))
    print('Loading epistasis degree table: {}'.format(degree_file), flush=True)
    df = pd.read_csv(degree_file, sep='\t', index_col=False, header=0)
    if df.shape[0] == 0:
        raise ValueError('Epistasis degree file is empty: {}'.format(degree_file))
    profile_map = _resolve_epistasis_profile_map_from_df(df=df, g=g, num_site=num_site)
    if len(profile_map) == 0:
        expected = [
            'epistasis_contact_degree(_z)',
            'epistasis_contact_proximity(_z)',
            'contact_degree(_z)',
            'contact_proximity(_z)',
            'epistasis_degree(_z)',
            'epistasis_proximity(_z)',
        ]
        txt = 'No epistasis site profile column was found in {}. Expected one of: {}.'
        raise ValueError(txt.format(degree_file, ','.join(expected)))
    selected_metric = str(g.get('epistasis_site_metric', 'auto')).strip().lower()
    out, resolved_metric = _select_epistasis_profile(
        profile_map=profile_map,
        metric=selected_metric,
        float_tol=float(g['float_tol']),
    )
    print(
        'Epistasis site metric: requested={}, resolved={} (source=file)'.format(
            selected_metric,
            resolved_metric,
        ),
        flush=True,
    )
    g['epistasis_site_metric_resolved'] = resolved_metric
    return np.asarray(out, dtype=np.float64).reshape(-1)


def _get_default_epistasis_branch_ids(g):
    branch_ids = [int(bid) for bid in g.get('sub_branches', [])]
    if len(branch_ids) == 0:
        branch_ids = [
            int(ete.get_prop(node, "numerical_label"))
            for node in g['tree'].traverse()
            if not ete.is_root(node)
        ]
    if len(branch_ids) == 0:
        raise ValueError('No non-root branch IDs were available for epistasis structure search.')
    return np.array(sorted(list(set(branch_ids))), dtype=np.int64)


def _compute_epistasis_degree_from_structure(g, num_site):
    epistasis_pdb = str(g.get('epistasis_pdb', '')).strip()
    if epistasis_pdb == '':
        return None
    from csubst import parser_biodb
    from csubst import parser_pymol
    print('Preparing epistasis structure degree from: {}'.format(epistasis_pdb), flush=True)
    site_index_alignment = parser_misc.get_site_index_alignment(g=g, expected_num_site=num_site)
    df = pd.DataFrame({
        'site': np.arange(num_site, dtype=np.int64),
        'codon_site_alignment': site_index_alignment + 1,
    })
    g_pdb = dict(g)
    g_pdb['pdb'] = epistasis_pdb
    g_pdb['database'] = g['epistasis_database']
    g_pdb['database_evalue_cutoff'] = g['epistasis_database_evalue_cutoff']
    g_pdb['database_minimum_identity'] = g['epistasis_database_minimum_identity']
    g_pdb['database_timeout'] = g['epistasis_database_timeout']
    g_pdb['pymol_max_num_chain'] = g['epistasis_pymol_max_num_chain']
    g_pdb['mafft_exe'] = g['epistasis_mafft_exe']
    g_pdb['mafft_op'] = g['epistasis_mafft_op']
    g_pdb['mafft_ep'] = g['epistasis_mafft_ep']
    g_pdb['branch_ids'] = _get_default_epistasis_branch_ids(g=g)
    if str(g_pdb['pdb']).strip().lower() == 'besthit':
        g_pdb = parser_biodb.pdb_sequence_search(g_pdb)
    if g_pdb.get('pdb', None) in [None, '']:
        raise ValueError('No structure could be resolved for epistasis degree generation.')
    parser_pymol.initialize_pymol(pdb_id=g_pdb['pdb'])
    epistasis_user_alignment = str(g.get('epistasis_user_alignment', '')).strip()
    if epistasis_user_alignment != '':
        print('Using epistasis user alignment for structure mapping: {}'.format(epistasis_user_alignment), flush=True)
        df = parser_pymol.add_coordinate_from_user_alignment(df=df, user_alignment=epistasis_user_alignment)
    else:
        g_pdb['mafft_add_fasta'] = os.path.abspath('tmp.csubst.epistasis.pdb_seq.fa')
        parser_pymol.write_mafft_alignment(g=g_pdb)
        df = parser_pymol.add_coordinate_from_mafft_map(df=df, mafft_map_file='tmp.csubst.pdb_seq.fa.map')
    df = parser_pymol.add_pdb_residue_numbering(df=df)
    df = parser_pymol.add_contact_degree_from_structure(
        df=df,
        distance_cutoff=float(g['epistasis_contact_distance']),
    )
    degree_outfile = os.path.abspath(g['epistasis_degree_outfile'])
    degree_cols = [
        'site',
        'codon_site_alignment',
        'epistasis_contact_chain',
        'epistasis_contact_residue',
        'epistasis_contact_degree',
        'epistasis_contact_degree_z',
        'epistasis_contact_proximity',
        'epistasis_contact_proximity_z',
    ]
    degree_cols_existing = [col for col in degree_cols if col in df.columns]
    df.loc[:, degree_cols_existing].to_csv(
        degree_outfile,
        sep='\t',
        index=False,
        float_format=g['float_format'],
    )
    print('Writing epistasis structure degree table: {}'.format(degree_outfile), flush=True)
    profile_map = _resolve_epistasis_profile_map_from_df(df=df, g=g, num_site=num_site)
    if len(profile_map) == 0:
        raise ValueError('No epistasis profile could be generated from structure mapping output.')
    selected_metric = str(g.get('epistasis_site_metric', 'auto')).strip().lower()
    degree_internal, resolved_metric = _select_epistasis_profile(
        profile_map=profile_map,
        metric=selected_metric,
        float_tol=float(g['float_tol']),
    )
    print(
        'Epistasis site metric: requested={}, resolved={} (source=structure)'.format(
            selected_metric,
            resolved_metric,
        ),
        flush=True,
    )
    g['epistasis_site_metric_resolved'] = resolved_metric
    if degree_internal.shape[0] != num_site:
        txt = 'Unexpected epistasis degree length. Expected {}, observed {}.'
        raise ValueError(txt.format(num_site, degree_internal.shape[0]))
    return degree_internal


def _prepare_epistasis_configuration(g, ON_tensor, OS_tensor):
    if not bool(g.get('epistasis_requested', False)):
        g['epistasis_enabled'] = False
        return g
    if str(g.get('omegaC_method', '')).strip().lower() != 'modelfree':
        raise ValueError('--epistasis_beta should be used with --omegaC_method "modelfree".')
    num_site = int(ON_tensor.shape[1])
    degree_internal = _load_epistasis_degree_from_file(g=g, num_site=num_site)
    if degree_internal is None:
        degree_internal = _compute_epistasis_degree_from_structure(g=g, num_site=num_site)
    if degree_internal is None:
        txt = 'Epistasis correction requires --epistasis_degree_file or --epistasis_pdb when --epistasis_beta is active.'
        raise ValueError(txt)
    g['epistasis_site_degree_internal'] = np.asarray(degree_internal, dtype=np.float64).reshape(-1)
    if g['epistasis_site_degree_internal'].shape[0] != num_site:
        txt = 'epistasis_site_degree_internal length ({}) did not match site axis ({}).'
        raise ValueError(txt.format(g['epistasis_site_degree_internal'].shape[0], num_site))
    g = omega.prepare_epistasis(g=g, ON_tensor=ON_tensor, OS_tensor=OS_tensor)
    return g


def main_analyze(g):
    start = time.time()
    print("Reading and parsing input files.", flush=True)
    g['current_arity'] = 2
    g['codon_table'] = genetic_code.get_codon_table(ncbi_id=g['genetic_code'])
    g = tree.read_treefile(g)
    g = parser_misc.generate_intermediate_files(g)
    g = parser_misc.annotate_tree(g)
    g = parser_misc.read_input(g)
    g = foreground.get_foreground_branch(g)
    g = foreground.get_marginal_branch(g)
    g = parser_misc.resolve_state_loading(g)
    g = parser_misc.prep_state(g)
    loaded_branch_ids = g.get('state_loaded_branch_ids', None)
    if loaded_branch_ids is not None:
        txt = 'Selective state loading active: writing alignments only for loaded nodes ({:,}).'
        print(txt.format(loaded_branch_ids.shape[0]), flush=True)
    sequence.write_alignment('csubst_alignment_codon.fa', mode='codon', g=g, branch_ids=loaded_branch_ids)
    sequence.write_alignment('csubst_alignment_aa.fa', mode='aa', g=g, branch_ids=loaded_branch_ids)
    if str(g.get('nonsyn_recode', 'no')).strip().lower() == '3di20':
        sequence.write_alignment('csubst_alignment_3di.fa', mode='nsy', g=g, branch_ids=loaded_branch_ids)
    g = combination.get_dep_ids(g)
    ON_tensor = substitution.get_substitution_tensor(state_tensor=g['state_nsy'], mode='asis', g=g, mmap_attr='N')
    ON_tensor = substitution.apply_min_sub_pp(g, ON_tensor)
    sub_branches = np.where(substitution.get_branch_sub_counts(ON_tensor) != 0)[0].tolist()
    OS_tensor = substitution.get_substitution_tensor(state_tensor=g['state_cdn'], mode='syn', g=g, mmap_attr='S')
    OS_tensor = substitution.apply_min_sub_pp(g, OS_tensor)
    sub_branches = list(set(sub_branches).union(set(np.where(substitution.get_branch_sub_counts(OS_tensor) != 0)[0].tolist())))
    g['sub_branches'] = sub_branches
    g = tree.rescale_branch_length(g, OS_tensor, ON_tensor)
    id_combinations = None
    OS_total = substitution.get_total_substitution(OS_tensor)
    ON_total = substitution.get_total_substitution(ON_tensor)
    num_branch = g['num_node'] - 1
    num_site = OS_tensor.shape[1]
    print('Synonymous substitutions / tree = {:,.1f}'.format(OS_total), flush=True)
    print('Nonsynonymous substitutions / tree = {:,.1f}'.format(ON_total), flush=True)
    print('Synonymous substitutions / branch = {:,.1f}'.format(OS_total / num_branch), flush=True)
    print('Nonsynonymous substitutions / branch = {:,.1f}'.format(ON_total / num_branch), flush=True)
    print('Synonymous substitutions / site = {:,.1f}'.format(OS_total / num_site), flush=True)
    print('Nonsynonymous substitutions / site = {:,.1f}'.format(ON_total / num_site), flush=True)
    elapsed_time = int(time.time() - start)
    print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['bs']):
        start = time.time()
        print("Generating bs table", flush=True)
        bs = substitution.get_bs(OS_tensor, ON_tensor)
        bs = _remap_site_column_to_alignment(df=bs, g=g, column_name='site')
        if bool(g.get('drop_invariant_tip_sites', False)):
            bs = parser_misc.expand_site_axis_table_to_alignment(
                df=bs,
                g=g,
                site_col='site',
                group_cols=['branch_id'],
                site_is_one_based=False,
            )
        bs = table.sort_branch_ids(df=bs)
        bs.to_csv("csubst_bs.tsv", sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
        txt = 'Memory consumption of bs table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(bs.values.nbytes/1024/1024, bs.values.dtype), flush=True)
        del bs
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['s']) | (g['cb']):
        start = time.time()
        print("Generating s table", flush=True)
        sOS = substitution.get_s(OS_tensor, attr='S')
        sON = substitution.get_s(ON_tensor, attr='N')
        s = table.merge_tables(sOS, sON)
        g = substitution.get_sub_sites(g, sOS, sON, state_tensor=g['state_cdn'])
        g = _prepare_epistasis_configuration(g=g, ON_tensor=ON_tensor, OS_tensor=OS_tensor)
        del sOS, sON
        if (g['s']):
            s = _remap_site_column_to_alignment(df=s, g=g, column_name='site')
            if bool(g.get('drop_invariant_tip_sites', False)):
                s = parser_misc.expand_site_axis_table_to_alignment(
                    df=s,
                    g=g,
                    site_col='site',
                    group_cols=[],
                    site_is_one_based=False,
                )
            s.to_csv("csubst_s.tsv", sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
        txt = 'Memory consumption of s table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(s.values.nbytes/1024/1024, s.values.dtype), flush=True)
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['omegaC_method']!='submodel'):
        g['state_cdn'] = None
        g['state_pep'] = None
        g['state_nsy'] = None

    if (g['b']) | (g['cb']):
        start = time.time()
        print("Generating b table", flush=True)
        bOS = substitution.get_b(g=g, sub_tensor=OS_tensor, attr='S', sitewise=False)
        bON = substitution.get_b(g=g, sub_tensor=ON_tensor, attr='N', sitewise=True)
        b = table.merge_tables(bOS, bON)
        b = _annotate_branch_length_column(b=b, tree_obj=g['tree'])
        txt = 'Number of {} patterns among {:,} branches={:,}, min={:,.1f}, max={:,.1f}'
        for key in ['S_sub', 'N_sub']:
            p = b.loc[:, key].drop_duplicates().values
            print(txt.format(key, b.shape[0], p.shape[0], p.min(), p.max()), flush=True)
        del bOS, bON
        b = foreground.annotate_b_foreground(b, g)
        g['branch_table'] = b
        if (g['b']):
            b_column_original = b.columns.tolist()
            b.columns = b.columns.str.replace('_PLACEHOLDER', '')
            b.to_csv("csubst_b.tsv", sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
            b.columns = b_column_original
        txt = 'Memory consumption of b table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(b.values.nbytes/1024/1024, b.values.dtype), flush=True)
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['cs']):
        start = time.time()
        print("Generating cs table", flush=True)
        if id_combinations is None:
            g,id_combinations = combination.get_node_combinations(g=g, exhaustive=True, arity=g['current_arity'], check_attr="name")
        reducer_OS_tensor = substitution.get_reducer_sub_tensor(sub_tensor=OS_tensor, g=g, label='csOS')
        reducer_ON_tensor = substitution.get_reducer_sub_tensor(sub_tensor=ON_tensor, g=g, label='csON')
        csOS = substitution.get_cs(id_combinations, reducer_OS_tensor, attr='S')
        csON = substitution.get_cs(id_combinations, reducer_ON_tensor, attr='N')
        cs = table.merge_tables(csOS, csON)
        del csOS, csON
        cs = _remap_site_column_to_alignment(df=cs, g=g, column_name='site')
        if bool(g.get('drop_invariant_tip_sites', False)):
            cs = parser_misc.expand_site_axis_table_to_alignment(
                df=cs,
                g=g,
                site_col='site',
                group_cols=[],
                site_is_one_based=False,
            )
        cs.to_csv("csubst_cs.tsv", sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
        txt = 'Memory consumption of cs table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(cs.values.nbytes/1024/1024, cs.values.dtype), flush=True)
        del cs
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['cbs']):
        start = time.time()
        print("Generating cbs table", flush=True)
        if id_combinations is None:
            g,id_combinations = combination.get_node_combinations(g=g, exhaustive=True, arity=g['current_arity'], check_attr="name")
        cbsOS = substitution.get_cbs(id_combinations, OS_tensor, attr='S', g=g)
        cbsON = substitution.get_cbs(id_combinations, ON_tensor, attr='N', g=g)
        cbs = table.merge_tables(cbsOS, cbsON)
        del cbsOS, cbsON
        cbs = _remap_site_column_to_alignment(df=cbs, g=g, column_name='site')
        if bool(g.get('drop_invariant_tip_sites', False)):
            cbs_group_cols = [col for col in cbs.columns.tolist() if str(col).startswith('branch_id_')]
            cbs = parser_misc.expand_site_axis_table_to_alignment(
                df=cbs,
                g=g,
                site_col='site',
                group_cols=cbs_group_cols,
                site_is_one_based=False,
            )
        cbs.to_csv("csubst_cbs.tsv", sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
        txt = 'Memory consumption of cbs table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(cbs.values.nbytes/1024/1024, cbs.values.dtype), flush=True)
        del cbs
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['cb']):
        g['df_cb_stats_main'] = pd.DataFrame()
        g,cb = cb_search(g, b, OS_tensor, ON_tensor, id_combinations, write_cb=True)
        #if (g['fg_clade_permutation']>0):
        #    g = foreground.clade_permutation(cb, g)
        #del cb
        g['df_cb_stats_main'] = table.sort_cb_stats(cb_stats=g['df_cb_stats_main'])
        print('Writing csubst_cb_stats.tsv', flush=True)
        column_original = g['df_cb_stats_main'].columns
        g['df_cb_stats_main'].columns = pd.Index(
            [str(col).replace('_PLACEHOLDER', '') for col in column_original]
        )
        g['df_cb_stats_main'].to_csv('csubst_cb_stats.tsv', sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
        g['df_cb_stats_main'].columns = column_original

    tmp_files = [f for f in os.listdir() if f.startswith('tmp.csubst.')]
    _ = [os.remove(ts) for ts in tmp_files]
