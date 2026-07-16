"""Shared type declarations for the analysis context mapping.

The command pipeline intentionally passes one mutable mapping between stages.
Keeping its commonly used keys here gives editors and static checks a useful
contract while the runtime representation remains backward-compatible.
"""

from typing import Any, TypedDict

import numpy as np
import pandas as pd


class AnalysisConfig(TypedDict, total=False):
    # Input and output
    alignment_file: str
    rooted_tree_file: str
    path_iqtree_state: str
    path_iqtree_rate: str
    output_dir: str
    output_prefix: str
    infile_type: str
    input_data_type: str

    # Numerical policy
    float_type: Any
    float_format: str
    float_tol: float
    threads: int
    current_arity: int
    max_arity: int

    # Model configuration
    expectation_method: str
    expected_state_backend: str
    nonsyn_recode: str
    ml_anc: bool
    codon_orders: np.ndarray
    amino_acid_orders: list[str]
    synonymous_indices: dict[str, list[int]]
    max_synonymous_size: int
    equilibrium_frequency: np.ndarray
    empirical_eq_freq: np.ndarray
    instantaneous_codon_rate_matrix: np.ndarray
    instantaneous_nsy_rate_matrix: np.ndarray
    iqtree_rate_values: np.ndarray

    # Tree and state storage
    tree: Any
    num_node: int
    num_input_site: int
    num_input_state: int
    state_cdn: np.ndarray | None
    state_pep: np.ndarray | None
    state_nsy: np.ndarray | None
    state_loaded_branch_ids: np.ndarray | None

    # Sparse reducers and analysis tables
    EN_reducer: dict[str, Any]
    ES_reducer: dict[str, Any]
    sub_branches: list[int]
    branch_table: pd.DataFrame
    df_cb_stats: pd.DataFrame
    df_cb_stats_main: pd.DataFrame

    # Output switches
    b: bool
    bs: bool
    cb: bool
    cbs: bool
    cs: bool
    s: bool
    branch_dist: bool
    drop_invariant_tip_sites: bool
    output_stats: Any

    # Internal lifetime/performance controls
    _release_state_after_expected_reducer: bool
    _cbs_stream_target_bytes: int
