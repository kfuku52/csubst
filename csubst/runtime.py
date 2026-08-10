import hashlib
import os
import shutil
import subprocess
import tempfile
from collections.abc import MutableMapping
from contextlib import contextmanager
from types import MappingProxyType


_RUN_TMPDIR_ENV = "CSUBST_RUN_TMPDIR"
_RUN_TMPDIR_PREFIX = ".csubst_tmp_"
_DEFAULT_OUTPUT_DIR = "."
_DEFAULT_OUTPUT_PREFIX = "csubst"
_DEFAULT_IQTREE_OUTDIR = "csubst_iqtree"
_DEFAULT_SITE_LOG_DIR = "csubst_sites"
_MISSING = object()
_NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_native_threads(num_threads=1):
    """Set native-library thread limits before NumPy/SciPy are imported."""

    num_threads = int(num_threads)
    if num_threads < 1:
        raise ValueError("--blas_threads should be >= 1.")
    for variable_name in _NATIVE_THREAD_ENV_VARS:
        os.environ[variable_name] = str(num_threads)
    return num_threads


class RunContext(MutableMapping):
    def __init__(self, config=None, state=None):
        self._config = dict(config or {})
        self._state = dict(state or {})

    @property
    def config(self):
        return MappingProxyType(self._config)

    @property
    def runtime_state(self):
        return self._state

    def __getitem__(self, key):
        if key in self._state:
            return self._state[key]
        return self._config[key]

    def __setitem__(self, key, value):
        self._state[key] = value

    def __delitem__(self, key):
        if key in self._state:
            del self._state[key]
            return
        if key in self._config:
            del self._config[key]
            return
        raise KeyError(key)

    def __iter__(self):
        seen = set()
        for key in self._state.keys():
            seen.add(key)
            yield key
        for key in self._config.keys():
            if key not in seen:
                yield key

    def __len__(self):
        return len(set(self._state.keys()).union(set(self._config.keys())))

    def __contains__(self, key):
        return (key in self._state) or (key in self._config)

    def copy(self):
        out = dict(self._config)
        out.update(self._state)
        return out

    def pop(self, key, default=_MISSING):
        if key in self._state:
            return self._state.pop(key)
        if key in self._config:
            return self._config.pop(key)
        if default is not _MISSING:
            return default
        raise KeyError(key)

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        self[key] = default
        return default


def ensure_run_context(g):
    if isinstance(g, RunContext):
        return g
    return RunContext(config=g)


def _normalize_outdir(value):
    if value is None:
        value = _DEFAULT_OUTPUT_DIR
    outdir = str(value).strip()
    if outdir == "":
        raise ValueError("--outdir should be non-empty.")
    return os.path.abspath(outdir)


def _normalize_output_prefix(value):
    if value is None:
        value = _DEFAULT_OUTPUT_PREFIX
    prefix = str(value).strip()
    if prefix == "":
        raise ValueError("--output_prefix should be non-empty.")
    if prefix != os.path.basename(prefix):
        raise ValueError("--output_prefix should be a file-name stem, not a path.")
    return prefix


def ensure_output_layout(g, create_dir=False):
    outdir = _normalize_outdir(g.get("outdir", _DEFAULT_OUTPUT_DIR))
    prefix = _normalize_output_prefix(g.get("output_prefix", _DEFAULT_OUTPUT_PREFIX))
    log_file = str(g.get("log_file", "")).strip()
    if log_file == "":
        log_file = os.path.join(outdir, prefix + ".log")
    elif not os.path.isabs(log_file):
        log_file = os.path.join(outdir, log_file)
    log_file = os.path.abspath(log_file)
    if create_dir:
        os.makedirs(outdir, exist_ok=True)
        log_dir = os.path.dirname(log_file)
        if log_dir != "":
            os.makedirs(log_dir, exist_ok=True)
    g["outdir"] = outdir
    g["output_prefix"] = prefix
    g["log_file"] = log_file
    return g


def _normalize_iqtree_outdir(value):
    if value is None:
        value = _DEFAULT_IQTREE_OUTDIR
    outdir = str(value).strip()
    if outdir == "":
        raise ValueError("--iqtree_outdir should be non-empty.")
    return os.path.abspath(outdir)


def ensure_iqtree_layout(g, create_dir=False):
    iqtree_outdir = _normalize_iqtree_outdir(g.get("iqtree_outdir", _DEFAULT_IQTREE_OUTDIR))
    if create_dir:
        os.makedirs(iqtree_outdir, exist_ok=True)
    g["iqtree_outdir"] = iqtree_outdir
    return g


def default_site_log_path(base_dir=None, create_dir=False):
    if base_dir is None:
        base_dir = os.getcwd()
    site_log_dir = os.path.abspath(os.path.join(str(base_dir), _DEFAULT_SITE_LOG_DIR))
    if create_dir:
        os.makedirs(site_log_dir, exist_ok=True)
    return os.path.join(site_log_dir, _DEFAULT_OUTPUT_PREFIX + ".log")




def _get_iqtree_prefix_key(alignment_file, base_dir=None):
    alignment_txt = str(alignment_file).strip()
    if alignment_txt == "":
        raise ValueError("--alignment_file should be non-empty to infer IQ-TREE outputs.")
    if base_dir is None:
        base_dir = os.getcwd()
    base_dir_abs = os.path.abspath(os.path.expanduser(str(base_dir)))
    expanded_alignment = os.path.expanduser(alignment_txt)
    if os.path.isabs(expanded_alignment):
        alignment_abs = os.path.abspath(expanded_alignment)
    else:
        alignment_abs = os.path.abspath(os.path.join(base_dir_abs, expanded_alignment))
    base_name = os.path.basename(alignment_abs)
    safe_base = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base_name)
    if safe_base == "":
        safe_base = "alignment"
    try:
        relative_alignment = os.path.relpath(alignment_abs, start=base_dir_abs)
    except ValueError:
        relative_alignment = alignment_abs
    if (relative_alignment == os.pardir) or relative_alignment.startswith(os.pardir + os.sep):
        identity = os.path.normcase(alignment_abs)
    else:
        identity = os.path.normcase(relative_alignment)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return "{}.{}".format(safe_base, digest)


def infer_iqtree_output_prefix(
    alignment_file,
    iqtree_outdir=None,
    base_dir=None,
    create_dir=False,
):
    resolved_outdir = _normalize_iqtree_outdir(iqtree_outdir)
    if create_dir:
        os.makedirs(resolved_outdir, exist_ok=True)
    prefix_key = _get_iqtree_prefix_key(alignment_file=alignment_file, base_dir=base_dir)
    return os.path.join(resolved_outdir, prefix_key)


def output_path(g, suffix, separator="_", create_dir=False):
    outdir = _normalize_outdir(g.get("outdir", _DEFAULT_OUTPUT_DIR))
    prefix = _normalize_output_prefix(g.get("output_prefix", _DEFAULT_OUTPUT_PREFIX))
    suffix_txt = str(suffix).strip()
    if suffix_txt == "":
        file_name = prefix
    elif separator is None:
        file_name = suffix_txt
    else:
        file_name = prefix + str(separator) + suffix_txt
    path = os.path.join(outdir, file_name)
    if create_dir:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def resolve_user_output_path(g, path, default_suffix=None, separator="_", create_dir=False):
    path_txt = "" if path is None else str(path).strip()
    if path_txt == "":
        if default_suffix is None:
            raise ValueError("A default output suffix is required when path is empty.")
        resolved = output_path(g=g, suffix=default_suffix, separator=separator, create_dir=create_dir)
    elif os.path.isabs(path_txt):
        resolved = path_txt
    else:
        outdir = _normalize_outdir(g.get("outdir", _DEFAULT_OUTPUT_DIR))
        resolved = os.path.join(outdir, path_txt)
    resolved = os.path.abspath(resolved)
    if create_dir:
        parent = os.path.dirname(resolved)
        if parent != "":
            os.makedirs(parent, exist_ok=True)
    return resolved


def resolve_user_output_prefix(g, prefix, default_suffix=None, separator="_", create_dir=False):
    prefix_txt = "" if prefix is None else str(prefix).strip()
    if prefix_txt == "":
        if default_suffix is None:
            raise ValueError("A default output suffix is required when prefix is empty.")
        resolved = output_path(
            g=g,
            suffix=default_suffix,
            separator=separator,
            create_dir=create_dir,
        )
    elif os.path.isabs(prefix_txt):
        resolved = prefix_txt
    else:
        outdir = _normalize_outdir(g.get("outdir", _DEFAULT_OUTPUT_DIR))
        resolved = os.path.join(outdir, prefix_txt)
    resolved = os.path.abspath(resolved)
    if create_dir:
        parent = os.path.dirname(resolved)
        if parent != "":
            os.makedirs(parent, exist_ok=True)
    return resolved


def get_run_tempdir(create=False, base_dir=None):
    path = str(os.environ.get(_RUN_TMPDIR_ENV, "")).strip()
    if path != "":
        if create and (not os.path.isdir(path)):
            os.makedirs(path, exist_ok=True)
        return path
    if not create:
        return None
    if base_dir is None:
        configured_tmpdir = str(os.environ.get("CSUBST_TMPDIR", "")).strip()
        base_dir = os.path.abspath(os.path.expanduser(configured_tmpdir)) if configured_tmpdir else None
    if base_dir is not None:
        os.makedirs(base_dir, exist_ok=True)
    path = tempfile.mkdtemp(prefix=_RUN_TMPDIR_PREFIX, dir=base_dir)
    os.environ[_RUN_TMPDIR_ENV] = path
    return path


def temp_path(path):
    path_txt = str(path).strip()
    if path_txt == "":
        raise ValueError("Temporary path name should be non-empty.")
    if os.path.isabs(path_txt):
        return path_txt
    run_tmpdir = get_run_tempdir(create=False)
    if run_tmpdir is None:
        return path_txt
    return os.path.join(run_tmpdir, os.path.basename(path_txt))


def cleanup_run_tempdir():
    path = get_run_tempdir(create=False)
    if path is not None:
        shutil.rmtree(path, ignore_errors=True)
    os.environ.pop(_RUN_TMPDIR_ENV, None)


def cleanup_legacy_temp_artifacts(base_dir=None, prefix="tmp.csubst."):
    """Retained as a compatibility no-op.

    Prefix-based cleanup cannot establish file ownership and may delete user
    data or another CSUBST process's files. Current runs are cleaned exclusively
    through ``run_tempdir_context``.
    """
    return []


@contextmanager
def run_tempdir_context(base_dir=None):
    created = False
    if get_run_tempdir(create=False) is None:
        get_run_tempdir(create=True, base_dir=base_dir)
        created = True
    try:
        yield get_run_tempdir(create=False)
    finally:
        if created:
            cleanup_run_tempdir()


def replace_file_cross_device(source, destination):
    """Atomically materialize ``source`` at ``destination`` across filesystems."""
    source = os.path.abspath(str(source))
    destination = os.path.abspath(str(destination))
    if source == destination:
        return destination
    parent = os.path.dirname(destination) or "."
    os.makedirs(parent, exist_ok=True)
    fd, staged = tempfile.mkstemp(prefix=".{}.tmp.".format(os.path.basename(destination)), dir=parent)
    os.close(fd)
    try:
        shutil.copy2(source, staged)
        with open(staged, mode="rb") as handle:
            os.fsync(handle.fileno())
        os.replace(staged, destination)
        os.remove(source)
    except Exception:
        try:
            os.remove(staged)
        except FileNotFoundError:
            pass
        raise
    return destination


def run_subprocess_tee(command, cwd=None):
    """Run a child process while forwarding combined output through Python stdout."""
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="", flush=True)
        process.stdout.close()
    return int(process.wait())
