"""Search output ownership through direct and CLI-reserved entry points."""

import json

import pytest

from csubst import output_run


def test_search_run_rejects_directory_alias_without_archiving_active_outputs(tmp_path):
    root = tmp_path / 'real'
    root.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    first = {'outdir': str(root), 'output_prefix': 'csubst'}
    second = {'outdir': str(alias), 'output_prefix': 'csubst'}
    with output_run.search_run(first):
        table = root / 'csubst_cb_2.tsv'
        table.write_text('active result\n')
        record = root / 'csubst_search_run.json'
        original = record.read_bytes()
        with pytest.raises(TimeoutError, match='search output lock'):
            with output_run.search_run(second):
                pytest.fail('The second search acquired the active namespace.')
        assert table.read_text() == 'active result\n'
        assert record.read_bytes() == original
        assert not (root / '.csubst_search_history').exists()
    assert json.loads(record.read_text())['status'] == 'complete'


def test_cli_reservation_allows_one_run_but_rejects_nested_search(tmp_path):
    g = {'outdir': str(tmp_path), 'output_prefix': 'csubst'}
    with output_run.search_output_lock(g):
        with output_run.search_run(g):
            with pytest.raises(TimeoutError, match='search output lock'):
                with output_run.search_run(g):
                    pytest.fail('A nested search borrowed the CLI reservation.')
        # Completion of the analysis must not release the CLI's lock while
        # its log and output finalizers are still open.
        with pytest.raises(TimeoutError, match='search output lock'):
            with output_run.search_output_lock(g):
                pytest.fail('The CLI reservation was released before finalization.')
    with output_run.search_run(g):
        pass
    assert json.loads((tmp_path / 'csubst_search_run.json').read_text())['status'] == 'complete'
