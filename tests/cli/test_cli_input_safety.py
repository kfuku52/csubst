import json
import os
from pathlib import Path

import pytest

from cli_runner import run_csubst


@pytest.mark.parametrize('input_option,alias,parse_error', [
    ('alignment_file', alias, error)
    for alias in ('same', 'symlink', 'hardlink')
    for error in (None, 'unknown', 'early_type_error')
] + [
    (option, 'same', None)
    for option in ('rooted_tree_file', 'foreground', 'iqtree_state')
])
def test_log_collision_preserves_inputs(tmp_path, input_option, alias, parse_error):
    source = tmp_path / 'input.dat'
    original = b'>A\nATGGCT\n>B\nATGGCC\n'
    source.write_bytes(original)
    log = source if alias == 'same' else tmp_path / 'alias.log'
    if alias == 'symlink':
        log.symlink_to(source)
    elif alias == 'hardlink':
        os.link(source, log)
    args = ['doctor', '--' + input_option, str(source), '--outdir', str(tmp_path / 'out'),
            '--log_file', str(log)]
    if parse_error == 'unknown':
        args.append('--does_not_exist')
    elif parse_error == 'early_type_error':
        args[1:1] = ['--threads', 'invalid']
    result = run_csubst(args, tmp_path)
    assert result.returncode == 2
    assert source.read_bytes() == original
    assert log.read_bytes() == original
    assert '--log_file must not overwrite input' in result.stderr
    assert 'Traceback' not in result.stderr


def test_log_collision_protects_inferred_iqtree_input(tmp_path):
    from csubst.runtime import infer_iqtree_output_prefix

    alignment = tmp_path / 'input.fa'
    alignment.write_text('>A\nATGGCT\n')
    iqtree_dir = tmp_path / 'iqtree'
    iqtree_dir.mkdir()
    state_path = infer_iqtree_output_prefix(str(alignment), str(iqtree_dir), base_dir=tmp_path) + '.state'
    state = Path(state_path)
    state.write_text('keep precomputed ancestral states\n')
    result = run_csubst(['doctor', '--alignment_file', str(alignment),
                         '--iqtree_outdir', str(iqtree_dir), '--log_file', str(state)], tmp_path)
    assert result.returncode == 2
    assert state.read_text() == 'keep precomputed ancestral states\n'
    assert '--log_file must not overwrite input --iqtree_state' in result.stderr


def test_all_failed_benchmarks_exit_nonzero_and_keep_summary(tmp_path):
    result = run_csubst(['benchmark', '--alignment_file', 'missing.fa',
                         '--rooted_tree_file', 'missing.nwk'], tmp_path)
    assert result.returncode == 2
    summary = json.loads((tmp_path / 'csubst_benchmark/csubst_benchmark_summary.json').read_text())
    assert summary['counts'] == {'pass': 0, 'fail': 1}
    assert 'Benchmark failed' in result.stderr
