from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
import sys


_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VALID_PARALLEL_BACKENDS = ('auto', 'multiprocessing', 'threading', 'loky')
_DEFAULT_AUTO_BACKEND = 'multiprocessing'


def _has_shape(input_data):
    return hasattr(input_data, 'shape')


def _slice_chunk(input_data, start, size):
    if _has_shape(input_data):
        return input_data[start:start + size, ...]
    return input_data[start:start + size]


def _build_worker_pythonpath():
    # Keep worker imports aligned with the parent process package location.
    prioritized_paths = [_PACKAGE_ROOT]
    cwd = os.path.abspath(os.getcwd())
    if cwd not in prioritized_paths:
        prioritized_paths.append(cwd)
    if len(sys.path) > 0:
        path0 = sys.path[0]
        if path0 in ['', '.']:
            path0 = cwd
        else:
            path0 = os.path.abspath(path0)
        if path0 not in prioritized_paths:
            prioritized_paths.append(path0)
    existing = os.environ.get('PYTHONPATH', '')
    for path in existing.split(os.pathsep):
        if path == '':
            continue
        normalized = os.path.abspath(path)
        if normalized not in prioritized_paths:
            prioritized_paths.append(normalized)
    return os.pathsep.join(prioritized_paths)


def _set_spawn_worker_import_path(ctx):
    if ctx.get_start_method() not in ['spawn', 'forkserver']:
        return None, None, False
    old_pythonpath = os.environ.get('PYTHONPATH')
    old_sys_path = list(sys.path)
    new_pythonpath = _build_worker_pythonpath()
    prioritized_paths = [p for p in new_pythonpath.split(os.pathsep) if p != '']
    for path in reversed(prioritized_paths):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    os.environ['PYTHONPATH'] = new_pythonpath
    return old_pythonpath, old_sys_path, True


def _restore_import_path(old_pythonpath, old_sys_path, changed):
    if not changed:
        return
    sys.path[:] = old_sys_path
    if old_pythonpath is None:
        os.environ.pop('PYTHONPATH', None)
    else:
        os.environ['PYTHONPATH'] = old_pythonpath


def _get_num_items(input_data):
    if _has_shape(input_data):
        return int(input_data.shape[0])
    return int(len(input_data))


def resolve_n_jobs(num_items, threads):
    num_items = int(num_items)
    threads = int(threads)
    if threads < 1:
        raise ValueError('threads should be >= 1.')
    if num_items < 0:
        raise ValueError('num_items should be >= 0.')
    if num_items == 0:
        return 1
    return min(num_items, threads)


def resolve_parallel_backend(g, task='general'):
    backend = str(g.get('parallel_backend', 'auto')).lower()
    if backend not in _VALID_PARALLEL_BACKENDS:
        raise ValueError('parallel_backend should be one of auto, multiprocessing, threading, loky.')
    if backend != 'auto':
        return backend
    return _DEFAULT_AUTO_BACKEND


def resolve_joblib_backend(g, task='general'):
    # Backward-compatible alias retained while migration off joblib is in progress.
    return resolve_parallel_backend(g=g, task=task)


def resolve_chunk_factor(g, task='general'):
    if task == 'reducer':
        value = int(g.get('parallel_chunk_factor_reducer', 4))
    else:
        value = int(g.get('parallel_chunk_factor', 1))
    if value < 1:
        raise ValueError('parallel_chunk_factor should be >= 1.')
    return value


def _normalize_parallel_backend(backend):
    backend = str(backend).lower()
    if backend == 'auto':
        return _DEFAULT_AUTO_BACKEND
    if backend in _VALID_PARALLEL_BACKENDS:
        return backend
    raise ValueError('parallel_backend should be one of auto, multiprocessing, threading, loky.')


def run_starmap(func, args_iterable, n_jobs, backend='multiprocessing', chunksize=None):
    args_list = list(args_iterable)
    if len(args_list) == 0:
        return list()
    n_jobs = int(n_jobs)
    if n_jobs < 1:
        raise ValueError('n_jobs should be >= 1.')
    if n_jobs == 1:
        return [func(*args) for args in args_list]
    backend = _normalize_parallel_backend(backend)
    if backend == 'threading':
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(func, *args) for args in args_list]
            return [future.result() for future in futures]
    if chunksize is None:
        chunksize = max(1, len(args_list) // (n_jobs * 4))
    else:
        chunksize = int(chunksize)
        if chunksize < 1:
            raise ValueError('chunksize should be >= 1.')
    ctx = multiprocessing.get_context()
    old_pythonpath, old_sys_path, changed = _set_spawn_worker_import_path(ctx)
    try:
        with ctx.Pool(processes=n_jobs) as pool:
            return pool.starmap(func, args_list, chunksize=chunksize)
    finally:
        _restore_import_path(old_pythonpath, old_sys_path, changed)


def get_chunks(input_data, threads, chunk_factor=1):
    num_items = _get_num_items(input_data)
    if num_items == 0:
        return list(), list()
    chunk_factor = int(chunk_factor)
    if chunk_factor < 1:
        raise ValueError('chunk_factor should be >= 1.')
    workers = resolve_n_jobs(num_items=num_items, threads=threads)
    num_chunks = min(num_items, workers * chunk_factor)
    base = num_items // num_chunks
    rem = num_items % num_chunks
    chunk_sizes = [base for _ in range(num_chunks)]
    for i in range(rem):
        chunk_sizes[num_chunks - rem + i] += 1
    i = 0
    out_chunks = list()
    starts = list()
    for size in chunk_sizes:
        if size == 0:
            continue
        out_chunks.append(_slice_chunk(input_data=input_data, start=i, size=size))
        starts.append(i)
        i += size
    return out_chunks, starts
