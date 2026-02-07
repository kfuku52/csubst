def _get_num_items(input_data):
    if 'shape' in dir(input_data):
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


def resolve_joblib_backend(g, task='general'):
    backend = str(g.get('parallel_backend', 'auto')).lower()
    if backend not in ['auto', 'multiprocessing', 'threading', 'loky']:
        raise ValueError('parallel_backend should be one of auto, multiprocessing, threading, loky.')
    if backend != 'auto':
        return backend
    if task in ['reducer']:
        return 'multiprocessing'
    return 'multiprocessing'


def resolve_chunk_factor(g, task='general'):
    if task == 'reducer':
        value = int(g.get('parallel_chunk_factor_reducer', 4))
    else:
        value = int(g.get('parallel_chunk_factor', 1))
    if value < 1:
        raise ValueError('parallel_chunk_factor should be >= 1.')
    return value


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
        if 'shape' in dir(input_data):
            out_chunks.append(input_data[i:i + size, ...])
        else:
            out_chunks.append(input_data[i:i + size])
        starts.append(i)
        i += size
    return out_chunks, starts
