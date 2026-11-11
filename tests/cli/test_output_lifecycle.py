"""Exercise output ownership and finalization through the real CLI."""

import csv
import json
import os
from pathlib import Path
import urllib.request

import pytest
import requests

from cli_runner import run_csubst
from pymol_fakes import import_parser_pymol_with_fake_pymol


@pytest.fixture
def tiny_fit(tmp_path):
    """Six tips and two codons suffice for real CLI output-lifecycle checks."""
    import numpy as np
    from csubst import ete, main_simulate, tree

    phylogeny = tree.add_numerical_node_labels(ete.PhyloNode(
        '((A:0.1,B:0.1)X:0.1,((C:0.1,D:0.1)Y:0.1,(E:0.1,F:0.1)Z:0.1)W:0.1)R;',
        format=1,
    ))
    alignment = tmp_path / 'alignment.fa'
    alignment.write_text(''.join(
        '>{}\n{}\n'.format(name, 'GATAAG' if name in 'ACE' else 'GCTAAA')
        for name in 'ABCDEF'
    ))
    ancestors = tmp_path / 'ancestors.fa'
    ancestors.write_text(''.join('>{}\nGCTAAA\n'.format(name) for name in 'XYZWR'))
    files = main_simulate.write_true_asr_bundle(
        dict(tree=phylogeny, eq_freq=np.full(61, 1 / 61), substitution_model='GY+F',
             genetic_code=1, background_omega=0.2, kappa=2.),
        str(ancestors), str(tmp_path / 'fit'),
    )
    # The synthetic fit has one uniform rate category, as reported by IQ-TREE.
    with open(files['iqtree'], 'a') as report:
        report.write('Model of rate heterogeneity: Uniform\n')
    args = ['--alignment_file', str(alignment), '--rooted_tree_file', files['treefile']]
    for option, key in [('state', 'state'), ('tree', 'treefile'), ('rate', 'rate'),
                        ('iqtree', 'iqtree'), ('log', 'log')]:
        args += ['--iqtree_' + option, files[key]]
    ids = {node.name: int(ete.get_prop(node, 'numerical_label')) for node in phylogeny.traverse()}
    return args, ','.join(str(ids[name]) for name in 'AC')


def doctor_inputs(tmp_path):
    alignment = tmp_path / 'input.fa'
    alignment.write_text('>A\nATGGCT\n>B\nATGGCC\n')
    tree = tmp_path / 'tree.nwk'
    tree.write_text('(A:0.1,B:0.1);')
    return ['doctor', '--alignment_file', str(alignment), '--rooted_tree_file', str(tree),
            '--outdir', str(tmp_path), '--check_iqtree_exe', 'no']


@pytest.mark.parametrize('command,program', [('sites', 'mmseqs'), ('search', 'blastp')])
def test_local_besthit_failure_never_submits_query(command, program, tiny_fit, tmp_path, monkeypatch):
    from csubst import cli

    import_parser_pymol_with_fake_pymol(monkeypatch, pdb_fasta='')
    network_attempts = []

    def reject_network(*args, **kwargs):
        network_attempts.append((args, kwargs))
        raise AssertionError('Unexpected network request during local search')

    monkeypatch.setattr(requests.Session, 'request', reject_network)
    monkeypatch.setattr(urllib.request, 'urlopen', reject_network)

    # Simulate a tool that echoes its input on failure.
    backend = tmp_path / program
    query_source = '"$2"' if program == 'mmseqs' else ''
    backend.write_text('#!/bin/sh\ncat {} > "$TEST_QUERY"\ncat "$TEST_QUERY" >&2\nexit 17\n'.format(query_source))
    backend.chmod(0o700)
    recorded_query = tmp_path / 'query.received'
    monkeypatch.setenv('TEST_QUERY', str(recorded_query))
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'pdb.dbtype').touch()
    inputs, branch_ids = tiny_fit
    if command == 'sites':
        besthit_args = ['--pdb', 'besthit', '--branch_id', branch_ids, '--database', 'pdb,alphafold']
    else:
        besthit_args = [
            '--epistasis_pdb', 'besthit', '--epistasis_database', 'alphafold,pdb',
            '--expectation_method', 'urn', '--asrv', 'sn', '--epistasis_beta', '0.5',
            '--epistasis_context_file', 'context.tsv', '--epistasis_context_source', 'test fixture',
        ]
    log = tmp_path / 'command.log'
    with pytest.raises(SystemExit) as exc:
        cli.main([command, *inputs, *besthit_args, '--outdir', str(tmp_path / 'out'), '--log_file', str(log),
                  '--database_search', 'local', '--mmseqs_db', 'pdb', '--blastp_db', 'swissprot'])
    assert exc.value.code == 2
    assert recorded_query.read_text() in {'>query\nDK\n', '>query\nAK\n'}
    assert network_attempts == []
    logged = log.read_text()
    assert 'Local {} failed (exit status 17)'.format(program) in logged
    assert '>query' not in logged


def test_sites_reuses_local_searches_for_identical_queries_only_within_a_run(tiny_fit, tmp_path, monkeypatch):
    from csubst import cli, ete, local_sequence_search, tree

    import_parser_pymol_with_fake_pymol(monkeypatch, pdb_fasta='')
    inputs, _ = tiny_fit
    phylogeny = tree.add_numerical_node_labels(ete.PhyloNode(Path(inputs[3]).read_text().strip(), format=1))
    ids = {node.name: ete.get_prop(node, 'numerical_label') for node in phylogeny.traverse() if ete.is_leaf(node)}
    combinations = tmp_path / 'branch_pairs.tsv'
    combinations.write_text('branch_id_1\tbranch_id_2\tis_fg\n' + ''.join(
        '{}\t{}\tyes\n'.format(ids[first], ids[second]) for first, second in ('AC', 'BC', 'CD')
    ))
    calls = []

    def run_mmseqs(sequence, *_):
        calls.append(('mmseqs', sequence))
        return []

    def run_blastp(sequence, *_):
        calls.append(('blastp', sequence))
        return []

    monkeypatch.setattr(local_sequence_search, 'mmseqs_pdb_ids', run_mmseqs)
    monkeypatch.setattr(local_sequence_search, 'blastp_accessions', run_blastp)
    monkeypatch.chdir(tmp_path)
    args = [
        'sites', *inputs, '--branch_id', 'fg', '--cb_file', str(combinations), '--pdb', 'besthit',
        '--database_search', 'local', '--database', 'pdb,alphafold', '--mmseqs_db', 'pdb', '--blastp_db', 'swissprot',
        '--site_summary_plot', 'no', '--site_state_plot', 'no', '--tree_site_plot', 'no',
    ]
    expected_calls = [('mmseqs', 'DK'), ('blastp', 'DK'), ('mmseqs', 'AK'), ('blastp', 'AK')]
    for run in range(2):
        outdir = tmp_path / f'out-{run}'
        cli.main([*args, '--outdir', str(outdir)])
        assert len(list(outdir.glob('csubst.branch_id*/csubst.tsv'))) == 3
        assert calls == expected_calls
        calls.clear()


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
def test_sites_cli_preserves_untrimmed_cds_at_chimera_fasta_destination(tmp_path, tiny_fit):
    fit_args, branch_ids = tiny_fit
    source = tmp_path / 'sites' / ('csubst.branch_id' + branch_ids) / 'csubst_A.fasta'
    source.parent.mkdir(parents=True)
    source.write_text('>A\nGATAAG\n')
    original = source.read_bytes()
    result = run_csubst(['sites', *fit_args, '--branch_id', branch_ids,
                         '--outdir', 'sites', '--untrimmed_cds', str(source),
                         '--site_state_plot', 'no', '--site_summary_plot', 'no',
                         '--tree_site_plot', 'no'], tmp_path)
    assert result.returncode == 2
    assert 'must not overwrite input --untrimmed_cds' in result.stderr
    assert source.read_bytes() == original


@pytest.mark.slow
def test_search_rerun_archives_old_high_arity_and_reports_failure(tmp_path, tiny_fit):
    fit_args, _ = tiny_fit
    base = ['search', *fit_args,
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
