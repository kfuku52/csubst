"""Exercise output ownership and finalization through the real CLI."""

import csv
import json
import os
from pathlib import Path

import pytest

from cli_runner import run_csubst


def doctor_inputs(tmp_path):
    alignment = tmp_path / 'input.fa'
    alignment.write_text('>A\nATGGCT\n>B\nATGGCC\n')
    tree = tmp_path / 'tree.nwk'
    tree.write_text('(A:0.1,B:0.1);')
    return ['doctor', '--alignment_file', str(alignment), '--rooted_tree_file', str(tree),
            '--outdir', str(tmp_path), '--check_iqtree_exe', 'no']


@pytest.mark.parametrize('alias', ['same', 'symlink', 'hardlink'])
def test_doctor_output_cannot_overwrite_input(tmp_path, alias):
    args = doctor_inputs(tmp_path)
    source = tmp_path / 'input.fa'
    destination = tmp_path / 'csubst_doctor_summary.tsv'
    if alias == 'same':
        source.rename(destination)
        args[2] = str(destination)
        source = destination
    elif alias == 'symlink':
        destination.symlink_to(source)
    else:
        os.link(source, destination)
    original = source.read_bytes()
    result = run_csubst(args, tmp_path)
    assert result.returncode == 2
    assert source.read_bytes() == destination.read_bytes() == original
    assert 'must not overwrite input' in result.stderr
    assert not (tmp_path / 'csubst.log').exists()


@pytest.mark.parametrize('alias', ['same', 'symlink', 'hardlink'])
@pytest.mark.parametrize('parse_error', [False, True])
def test_log_output_collision_is_rejected_before_truncation(tmp_path, alias, parse_error):
    args = doctor_inputs(tmp_path)
    output = tmp_path / 'csubst_doctor_summary.json'
    output.write_text('{"previous": true}\n')
    log = output if alias == 'same' else tmp_path / 'alias.log'
    if alias == 'symlink':
        log.symlink_to(output)
    elif alias == 'hardlink':
        os.link(output, log)
    if parse_error:
        args[1:1] = ['--threads', 'invalid']
    result = run_csubst(args + ['--log_file', str(log)], tmp_path)
    assert result.returncode == 2
    assert json.loads(output.read_text()) == {'previous': True}
    assert 'must not overwrite log' in result.stderr


@pytest.mark.parametrize('fail', [False, True])
def test_manifest_matches_closed_log_even_on_failure(tmp_path, fail):
    args = doctor_inputs(tmp_path)
    if fail:
        args += ['--doctor_fail_level', 'warning']
    result = run_csubst(args, tmp_path)
    assert result.returncode == (2 if fail else 0)
    manifest = tmp_path / 'csubst_outputs.tsv'
    with manifest.open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    for row in rows:
        assert int(row['file_size_bytes']) == Path(row['output_path']).stat().st_size
    log = (tmp_path / 'csubst.log').read_text()
    assert ('Doctor checks found issues' if fail else 'CSUBST end:') in log


def test_search_log_cannot_destroy_existing_table_before_archival(tmp_path):
    output = tmp_path / 'csubst_cb_3.tsv'
    output.write_text('previous results\n')
    result = run_csubst(['search', '--alignment_file', 'missing.fa',
                         '--rooted_tree_file', 'missing.nwk', '--outdir', str(tmp_path),
                         '--log_file', str(output)], tmp_path)
    assert result.returncode == 2
    assert output.read_text() == 'previous results\n'


@pytest.mark.parametrize('parse_error', [False, True])
def test_concurrent_search_preserves_active_log_and_record(tmp_path, parse_error):
    from csubst import output_run

    out = tmp_path / 'out'
    out.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(out, target_is_directory=True)
    log = out / 'csubst.log'
    log.write_text('first search log\n')
    args = ['search', '--alignment_file', 'missing.fa', '--rooted_tree_file', 'missing.nwk',
            '--outdir', str(alias)]
    if parse_error:
        args += ['--threads', 'invalid']
    with output_run.search_run({'outdir': str(out), 'output_prefix': 'csubst'}):
        record = out / 'csubst_search_run.json'
        original_record = record.read_bytes()
        result = run_csubst(args, tmp_path)
        assert result.returncode == 2
        assert 'search output lock' in result.stderr
        assert log.read_text() == 'first search log\n'
        assert record.read_bytes() == original_record
        assert not (out / '.csubst_search_history').exists()


@pytest.mark.parametrize('destination', [
    'tree.nwk', 'alignment.fa.gz', 'foreground.txt',
    'csubst_iqtree/alignment.fa.gz.10f9ebf3efb5f43b.state',
    'csubst_iqtree/alignment.fa.gz.10f9ebf3efb5f43b.state.csubst-manifest.json',
])
def test_dataset_log_cannot_collide_with_generated_output(tmp_path, destination):
    result = run_csubst(['--log_file', destination, 'dataset', '--name', 'PGK',
                         '--force', 'yes'], tmp_path)
    assert result.returncode == 2
    assert 'must not overwrite log' in result.stderr
    assert not (tmp_path / destination).exists()
    assert not (tmp_path / 'tree.nwk').exists()
    assert not (tmp_path / 'alignment.fa.gz').exists()


@pytest.mark.parametrize('alias', ['symlink', 'hardlink'])
def test_dataset_preserves_existing_output_when_log_aliases_it(tmp_path, alias):
    tree = tmp_path / 'tree.nwk'
    tree.write_text('(A:1,B:1);\n')
    log = tmp_path / 'alias.log'
    if alias == 'symlink':
        log.symlink_to(tree)
    else:
        os.link(tree, log)
    result = run_csubst(['--log_file', str(log), 'dataset', '--name', 'PGK',
                         '--force', 'yes'], tmp_path)
    assert result.returncode == 2
    assert 'must not overwrite log' in result.stderr
    assert tree.read_text() == '(A:1,B:1);\n'
    assert not (tmp_path / 'alignment.fa.gz').exists()


def test_dataset_parse_error_does_not_append_to_generated_tree(tmp_path):
    tree = tmp_path / 'tree.nwk'
    tree.write_text('(A:1,B:1);\n')
    result = run_csubst(['--log_file', 'tree.nwk', 'dataset', '--force', 'invalid'], tmp_path)
    assert result.returncode == 2
    assert tree.read_text() == '(A:1,B:1);\n'


def test_dataset_preflight_does_not_import_numerical_libraries(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPROFILEIMPORTTIME', '1')
    result = run_csubst(['--log_file', 'tree.nwk', 'dataset'], tmp_path)
    assert result.returncode == 2
    assert 'must not overwrite log' in result.stderr
    imports = {line.rsplit('|', 1)[-1].strip().split('.')[0]
               for line in result.stderr.splitlines() if line.startswith('import time:')}
    assert not imports.intersection({'numpy', 'scipy'})


@pytest.mark.slow
def test_sites_cli_preserves_untrimmed_cds_at_chimera_fasta_destination(tmp_path):
    from csubst import sequence_io

    assert run_csubst(['dataset', '--name', 'PGK'], tmp_path).returncode == 0
    record = next(r for r in sequence_io.read_fasta_records(tmp_path / 'alignment.fa.gz')
                  if '-' not in r.sequence and 'N' not in r.sequence)
    source = tmp_path / 'sites' / 'csubst.branch_id23,51' / ('csubst_' + record.id + '.fasta')
    source.parent.mkdir(parents=True)
    sequence_io.write_fasta_records([record], source)
    original = source.read_bytes()
    result = run_csubst(['sites', '--alignment_file', 'alignment.fa.gz',
                         '--rooted_tree_file', 'tree.nwk', '--branch_id', '23,51',
                         '--outdir', 'sites', '--untrimmed_cds', str(source),
                         '--site_state_plot', 'no', '--site_summary_plot', 'no',
                         '--tree_site_plot', 'no'], tmp_path)
    assert result.returncode == 2
    assert 'must not overwrite input --untrimmed_cds' in result.stderr
    assert source.read_bytes() == original


@pytest.mark.slow
def test_search_rerun_archives_old_high_arity_and_reports_failure(tmp_path):
    assert run_csubst(['dataset', '--name', 'PGK'], tmp_path).returncode == 0
    base = ['search', '--alignment_file', 'alignment.fa.gz', '--rooted_tree_file', 'tree.nwk',
            '--max_arity', '3', '--max_combination', '10', '--output_stat', 'any2spe']
    first = run_csubst(base + ['--cutoff_stat', 'OCNany2spe,0'], tmp_path)
    assert first.returncode == 0, first.stderr
    out = tmp_path / 'csubst_search'
    old = (out / 'csubst_cb_3.tsv').read_bytes()
    unrelated = out / 'notes.txt'
    unrelated.write_text('keep this file')
    second = run_csubst(base + ['--cutoff_stat', 'OCNany2spe,1000000000'], tmp_path)
    assert second.returncode == 0, second.stderr
    assert not (out / 'csubst_cb_3.tsv').exists()
    record = json.loads((out / 'csubst_search_run.json').read_text())
    assert record['status'] == 'complete'
    assert 'csubst_cb_2.tsv' in record['outputs']
    assert 'csubst_cb_3.tsv' not in record['outputs']
    archived = next(out / p for p in record['archived_outputs'] if p.endswith('csubst_cb_3.tsv'))
    assert archived.read_bytes() == old
    assert unrelated.read_text() == 'keep this file'
    failed = run_csubst(base + ['--alignment_file', 'missing.fa'], tmp_path)
    assert failed.returncode != 0
    record = json.loads((out / 'csubst_search_run.json').read_text())
    assert record['status'] == 'failed'
    assert not (out / 'csubst_cb_2.tsv').exists()
    assert unrelated.read_text() == 'keep this file'
