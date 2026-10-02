"""Preserve earlier search tables and record the state of the current run."""

import os
import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from csubst import output_safety, resource_cache, runtime


_reserved_search_lock: ContextVar[str | None] = ContextVar('csubst_reserved_search_lock', default=None)


def _acquire_search_lock(lock):
    return resource_cache.acquire_exclusive_lock(
        lock, lock_label='search output', poll_seconds=.1, timeout_seconds=.1,
    )


@contextmanager
def search_output_lock(g):
    """Reserve a search namespace before the CLI opens its log."""
    layout = runtime.ensure_output_layout(dict(g), create_dir=True)
    lock = resource_cache.resolve_path_lock_path(runtime.output_path(layout, 'search_run.json'))
    with _acquire_search_lock(lock):
        token = _reserved_search_lock.set(lock)
        try:
            yield
        finally:
            _reserved_search_lock.reset(token)


@contextmanager
def _claim_search_lock(lock):
    if _reserved_search_lock.get() != lock:
        with _acquire_search_lock(lock):
            yield
        return
    # Consume the CLI reservation for this run only. A nested or concurrent
    # search must still acquire the physical lock and cannot borrow ownership.
    token = _reserved_search_lock.set(None)
    try:
        yield
    finally:
        _reserved_search_lock.reset(token)


def _search_tables(g):
    root = Path(g['outdir'])
    prefix = re.escape(g['output_prefix'])
    pattern = re.compile(prefix + r'_(?:cb_[0-9]+|cb_stats|b|bs|s|cs|cbs)\.tsv$')
    return sorted(path for path in root.iterdir() if pattern.fullmatch(path.name))


@contextmanager
def search_run(g):
    layout = runtime.ensure_output_layout(dict(g), create_dir=True)
    root = Path(layout['outdir'])
    run_id = uuid.uuid4().hex
    manifest = runtime.output_path(layout, 'search_run.json')
    lock = resource_cache.resolve_path_lock_path(manifest)
    # Fail promptly instead of mixing tables from concurrent invocations.
    with _claim_search_lock(lock):
        previous = _search_tables(layout)
        if os.path.lexists(manifest):
            previous.append(Path(manifest))
        for path in previous:
            output_safety.validate_destination(path)
            if not path.is_file() or path.is_symlink():
                raise ValueError('Search output is not a regular file: {}'.format(path))
        history = root / '.csubst_search_history'
        if history.is_symlink():
            raise ValueError('Search history must not be a symbolic link: {}'.format(history))
        archive = history / run_id
        if previous:
            archive.mkdir(parents=True, exist_ok=False)
            for path in previous:
                os.replace(path, archive / path.name)
        record = {'schema_version': 1, 'run_id': run_id, 'status': 'running',
                  'archived_outputs': [str((archive / path.name).relative_to(root)) for path in previous],
                  'outputs': []}
        resource_cache.atomic_write_json(manifest, record)
        try:
            yield
        except BaseException as exc:
            record.update(status='failed', error=str(exc))
            raise
        else:
            record['status'] = 'complete'
        finally:
            record['outputs'] = [path.name for path in _search_tables(layout)]
            resource_cache.atomic_write_json(manifest, record)
