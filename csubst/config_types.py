"""Typed contracts for configuration and mutable analysis state.

The runtime still exposes one mapping for compatibility, but grouping keys by
lifetime makes stage boundaries reviewable and supports incremental typing.
"""

from typing import Any, TypedDict

import numpy as np
import pandas as pd


class InputConfig(TypedDict, total=False):
    alignment_file: str
    full_cds_alignment_file: str
    rooted_tree_file: str
    foreground: str
    path_iqtree_state: str
    path_iqtree_rate: str
    infile_type: str
    input_data_type: str
    genetic_code: int
    resource: str
    resource_cache_dir: str
    resource_lock_poll: float
    resource_lock_timeout: float
    no_download: bool
    verify: bool | None


class OutputConfig(TypedDict, total=False):
    outdir: str
    output_prefix: str
    log_file: str
    iqtree_outdir: str
    float_format: str
    float_digit: int


class NumericConfig(TypedDict, total=False):
    float_type: Any
    float_tol: float
    threads: int
    blas_threads: int
    random_seed: int
    current_arity: int
    max_arity: int
    min_arity: int
    min_sub_pp: float


class ModelConfig(TypedDict, total=False):
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


class AnalysisState(TypedDict, total=False):
    tree: Any
    num_node: int
    num_input_site: int
    num_input_state: int
    state_cdn: np.ndarray | None
    state_pep: np.ndarray | None
    state_nsy: np.ndarray | None
    state_loaded_branch_ids: np.ndarray | None
    EN_reducer: dict[str, Any]
    ES_reducer: dict[str, Any]
    sub_branches: list[int]
    branch_table: pd.DataFrame
    df_cb_stats: pd.DataFrame
    df_cb_stats_main: pd.DataFrame
    fg_ids: dict[str, np.ndarray]


class OutputSwitches(TypedDict, total=False):
    b: bool
    bs: bool
    cb: bool
    cbs: bool
    cs: bool
    s: bool
    branch_dist: bool
    drop_invariant_tip_sites: bool
    output_stats: Any


class InternalControls(TypedDict, total=False):
    _release_state_after_expected_reducer: bool
    _cbs_stream_target_bytes: int


class AnalysisConfig(
    InputConfig,
    OutputConfig,
    NumericConfig,
    ModelConfig,
    AnalysisState,
    OutputSwitches,
    InternalControls,
    total=False,
):
    """Compatibility view of the combined pipeline mapping."""
