"""Render observation eligibility at table boundaries, never in numeric tensors."""
import re

import numpy as np


def annotate(frame, g):
    tensors = g.get('_endpoint_tensors', {})
    masks = {kind: getattr(tensors.get(kind), 'eligible', None) for kind in ('S', 'N')}
    if all(mask is None for mask in masks.values()):
        return frame
    out = frame.copy()
    site_col = 'site' if 'site' in out else 'codon_site_alignment' if 'codon_site_alignment' in out else None
    sample = next(mask for mask in masks.values() if mask is not None)
    alignment = np.asarray(g.get('site_index_alignment', np.arange(sample.shape[1])))
    lookup = {int(site): i for i, site in enumerate(alignment)}
    sites = np.array([lookup.get(int(site) - (1 if site_col == 'codon_site_alignment' else 0), -1) if np.isfinite(site) else -1
                      for site in out[site_col]], dtype=int) if site_col else None
    branch_cols = [c for c in out if re.fullmatch(r'branch_id(?:_\d+)?', str(c))]
    for kind, mask in masks.items():
        if mask is None:
            continue
        columns = [c for c in out if c == kind + '_sub' or re.match(r'[OE]C' + kind, str(c))]
        def coverage(branches=None):
            if branches is None:
                values = mask.sum(axis=0)
            else:
                branches = np.asarray(branches, dtype=int)
                values = mask[branches].all(axis=0).astype(int)
            if sites is None:
                return int(values.sum())
            return np.where(sites >= 0, values[np.maximum(sites, 0)], 0)
        # Per-branch site columns (sites command).
        for col in list(out):
            match = re.fullmatch(kind + r'_sub_(\d+)', str(col))
            if not match:
                continue
            slot = 'branch_id_' + match[1]
            if slot in out:
                count: np.ndarray = np.zeros(len(out), dtype=int)
                for branch, rows in out.groupby(slot, sort=False).indices.items():
                    values = coverage([int(branch)])
                    count[rows] = values if sites is None else values[rows]
                out[kind + '_eligible_count_' + match[1]] = count
            elif sites is not None:
                count = coverage([int(match[1])])
                out[kind + '_eligible_' + match[1]] = count.astype(bool)
            else:
                continue
            out.loc[count == 0, col] = np.nan
        if not columns:
            continue
        if branch_cols:
            count = np.zeros(len(out), dtype=int)
            # Group duplicate branch combinations, keeping memory bounded by one site vector.
            for keys, rows in out.groupby(branch_cols, sort=False).indices.items():
                branches = keys if isinstance(keys, tuple) else (keys,)
                values = mask[np.asarray(branches, int)].all(axis=0)
                if sites is None:
                    count[rows] = values.sum()
                else:
                    selected = sites[rows]
                    count[rows] = np.where(selected >= 0, values[np.maximum(selected, 0)], False)
        else:
            count = coverage()
        out[kind + '_eligible_count'] = count
        out.loc[np.asarray(count == 0) if np.ndim(count) else np.full(len(out), count == 0), columns] = np.nan
        combo_cols = [c for c in columns if re.match(r'[OE]C' + kind, str(c))]
        combinations = g.get('report_combinations')
        if combinations is None and g.get('branch_ids') is not None:
            combinations = [g['branch_ids']]
        if combo_cols and not branch_cols and combinations is not None:
            joint_count: np.ndarray = sum((coverage(branches) for branches in combinations), np.zeros(len(out), dtype=int))
            out[kind + '_combination_eligible_count'] = joint_count
            out.loc[joint_count == 0, combo_cols] = np.nan
    if 'S_eligible_count' in out and 'N_eligible_count' in out:
        unavailable = (out['S_eligible_count'] == 0) | (out['N_eligible_count'] == 0)
        for kind in ('S', 'N'):
            key = kind + '_combination_eligible_count'
            if key in out:
                unavailable |= out[key] == 0
        ratio_cols = [c for c in out if str(c).startswith(('omegaC', 'dNC', 'dSC'))]
        out.loc[unavailable, ratio_cols] = np.nan
    return out
