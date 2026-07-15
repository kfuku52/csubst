import errno
import hashlib
import json
import os
import re
import shutil
import socket
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager


LOCK_FORMAT = "csubst-lock-v1"
MANIFEST_FORMAT = "csubst-resource-v1"
DEFAULT_LOCK_POLL_SECONDS = 5.0
DEFAULT_LOCK_TIMEOUT_SECONDS = 3600.0
DEFAULT_LOCK_HEARTBEAT_SECONDS = 60.0
DEFAULT_LOCK_STALE_SECONDS = 900.0
RESOURCE_MANIFEST_NAME = "csubst-resource.json"


def resolve_cache_dir(cache_dir=None):
    raw = "" if cache_dir is None else str(cache_dir).strip()
    if raw == "":
        raw = str(os.environ.get("CSUBST_CACHE_DIR", "")).strip()
    if raw == "":
        xdg_cache_home = str(os.environ.get("XDG_CACHE_HOME", "")).strip()
        if xdg_cache_home != "":
            raw = os.path.join(xdg_cache_home, "csubst")
        else:
            raw = os.path.join(os.path.expanduser("~"), ".cache", "csubst")
    return os.path.realpath(os.path.expanduser(raw))


def _sanitize_resource_name(value):
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return normalized or "resource"


def resolve_resource_lock_path(resource_id, cache_dir=None):
    resource_txt = str(resource_id).strip()
    digest = hashlib.sha256(resource_txt.encode("utf-8")).hexdigest()[:16]
    name = _sanitize_resource_name(resource_txt)[:80]
    return os.path.join(resolve_cache_dir(cache_dir), "locks", "{}-{}.lock".format(name, digest))


def resolve_path_lock_path(path, lock_label=None):
    target_path = os.path.abspath(os.path.expanduser(str(path)))
    parent = os.path.dirname(target_path) or "."
    base = os.path.basename(target_path)
    label = base if lock_label is None else str(lock_label)
    digest = hashlib.sha256(target_path.encode("utf-8")).hexdigest()[:16]
    name = _sanitize_resource_name(label)[:80]
    return os.path.join(parent, ".csubst-locks", "{}-{}.lock".format(name, digest))


def _assert_regular_file_or_absent(path, label="Path"):
    if not os.path.lexists(path):
        return
    if os.path.islink(path) or (not os.path.isfile(path)):
        raise IsADirectoryError("{} exists but is not a regular file: {}".format(label, path))


def _resolve_local_boot_id():
    path = "/proc/sys/kernel/random/boot_id"
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            value = handle.readline().strip()
    except OSError:
        return None
    return value or None


def _build_lock_metadata(owner_token):
    return {
        "format": LOCK_FORMAT,
        "owner_token": str(owner_token),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "boot_id": _resolve_local_boot_id(),
        "created_at": time.time(),
    }


def _read_lock_metadata(lock_path):
    try:
        with open(lock_path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    return value


def _describe_lock_owner(metadata):
    if not isinstance(metadata, dict):
        return "owner=unknown"
    parts = []
    if metadata.get("hostname"):
        parts.append("host={}".format(metadata["hostname"]))
    if metadata.get("pid"):
        parts.append("pid={}".format(metadata["pid"]))
    created_at = metadata.get("created_at")
    if isinstance(created_at, (float, int)):
        parts.append("created_at={:.0f}".format(created_at))
    return ", ".join(parts) if parts else "owner=unknown"


def _try_create_lock(lock_path, owner_token):
    _assert_regular_file_or_absent(lock_path, label="Lock path")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        payload = json.dumps(_build_lock_metadata(owner_token), sort_keys=True) + "\n"
        with os.fdopen(fd, mode="w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
        raise
    return True


def _is_process_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if getattr(exc, "errno", None) == errno.ESRCH:
            return False
        return True
    return True


def _is_same_host_and_boot(metadata):
    if not isinstance(metadata, dict):
        return False
    if metadata.get("hostname") != socket.gethostname():
        return False
    owner_boot_id = metadata.get("boot_id")
    local_boot_id = _resolve_local_boot_id()
    return bool(owner_boot_id and local_boot_id and owner_boot_id == local_boot_id)


def _lock_is_owned(lock_path, owner_token):
    metadata = _read_lock_metadata(lock_path)
    return isinstance(metadata, dict) and metadata.get("owner_token") == owner_token


def _start_heartbeat(lock_path, owner_token, interval_seconds):
    interval_seconds = float(interval_seconds)
    if interval_seconds <= 0:
        raise ValueError("heartbeat_seconds must be > 0.")
    stop_event = threading.Event()

    def heartbeat():
        while not stop_event.wait(interval_seconds):
            try:
                if not _lock_is_owned(lock_path, owner_token):
                    return
                os.utime(lock_path, None)
            except OSError:
                return

    thread = threading.Thread(target=heartbeat, name="csubst-resource-lock-heartbeat", daemon=True)
    thread.start()
    return stop_event, thread


def _release_lock(lock_path, owner_token, heartbeat_stop, heartbeat_thread, heartbeat_seconds):
    heartbeat_stop.set()
    heartbeat_thread.join(timeout=max(1.0, float(heartbeat_seconds) * 2.0))
    if not os.path.lexists(lock_path):
        return
    _assert_regular_file_or_absent(lock_path, label="Lock path")
    if _lock_is_owned(lock_path, owner_token):
        os.remove(lock_path)


def _break_stale_lock_if_needed(lock_path, lock_label, stale_seconds):
    if not os.path.lexists(lock_path):
        return False
    _assert_regular_file_or_absent(lock_path, label="{} lock path".format(lock_label))
    try:
        stat_before = os.stat(lock_path)
    except FileNotFoundError:
        return False
    metadata = _read_lock_metadata(lock_path)
    stale_reason = None
    try:
        owner_pid = int(metadata.get("pid")) if isinstance(metadata, dict) else None
    except (TypeError, ValueError):
        owner_pid = None
    same_host_process = _is_same_host_and_boot(metadata) and owner_pid is not None and owner_pid > 0
    if same_host_process:
        if not _is_process_alive(owner_pid):
            stale_reason = "same-host owner PID {} is not running".format(owner_pid)
    elif float(stale_seconds) > 0:
        heartbeat_age = time.time() - stat_before.st_mtime
        if heartbeat_age > float(stale_seconds):
            stale_reason = "heartbeat expired after {:.0f} sec".format(heartbeat_age)
    if stale_reason is None:
        return False
    try:
        stat_now = os.stat(lock_path)
    except FileNotFoundError:
        return False
    if (
        stat_before.st_ino != stat_now.st_ino
        or stat_before.st_size != stat_now.st_size
        or stat_before.st_mtime_ns != stat_now.st_mtime_ns
    ):
        return False
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        return False
    print("Removed stale {} lock: {} ({})".format(lock_label, lock_path, stale_reason), flush=True)
    return True


@contextmanager
def acquire_exclusive_lock(
    lock_path,
    lock_label="resource",
    poll_seconds=DEFAULT_LOCK_POLL_SECONDS,
    timeout_seconds=DEFAULT_LOCK_TIMEOUT_SECONDS,
    heartbeat_seconds=DEFAULT_LOCK_HEARTBEAT_SECONDS,
    stale_seconds=DEFAULT_LOCK_STALE_SECONDS,
):
    poll_seconds = float(poll_seconds)
    timeout_seconds = float(timeout_seconds)
    heartbeat_seconds = float(heartbeat_seconds)
    stale_seconds = float(stale_seconds)
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be > 0.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0.")
    if stale_seconds <= heartbeat_seconds:
        raise ValueError("stale_seconds must be greater than heartbeat_seconds.")
    lock_path = os.path.abspath(os.path.expanduser(str(lock_path)))
    _assert_regular_file_or_absent(lock_path, label="{} lock path".format(lock_label))
    lock_dir = os.path.dirname(lock_path)
    if os.path.exists(lock_dir) and (not os.path.isdir(lock_dir)):
        raise NotADirectoryError("Lock parent path exists but is not a directory: {}".format(lock_dir))
    os.makedirs(lock_dir, exist_ok=True)
    owner_token = uuid.uuid4().hex
    wait_start = time.monotonic()
    reported_wait = False
    while True:
        if _try_create_lock(lock_path, owner_token):
            heartbeat_stop, heartbeat_thread = _start_heartbeat(
                lock_path=lock_path,
                owner_token=owner_token,
                interval_seconds=heartbeat_seconds,
            )
            try:
                yield lock_path
            finally:
                _release_lock(
                    lock_path=lock_path,
                    owner_token=owner_token,
                    heartbeat_stop=heartbeat_stop,
                    heartbeat_thread=heartbeat_thread,
                    heartbeat_seconds=heartbeat_seconds,
                )
            return
        _assert_regular_file_or_absent(lock_path, label="{} lock path".format(lock_label))
        if _break_stale_lock_if_needed(lock_path, lock_label=lock_label, stale_seconds=stale_seconds):
            continue
        if not reported_wait:
            print(
                "Another process holds {}. Waiting for lock release: {} ({})".format(
                    lock_label,
                    lock_path,
                    _describe_lock_owner(_read_lock_metadata(lock_path)),
                ),
                flush=True,
            )
            reported_wait = True
        elapsed = time.monotonic() - wait_start
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                "Timed out after {:.1f} sec waiting for {} lock: {} ({})".format(
                    timeout_seconds,
                    lock_label,
                    lock_path,
                    _describe_lock_owner(_read_lock_metadata(lock_path)),
                )
            )
        time.sleep(min(poll_seconds, max(0.0, timeout_seconds - elapsed)))


def atomic_write_text(path, text, encoding="utf-8"):
    path = os.path.abspath(os.path.expanduser(str(path)))
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".{}.tmp.".format(os.path.basename(path)), dir=parent)
    try:
        with os.fdopen(fd, mode="w", encoding=encoding) as handle:
            handle.write(str(text))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise
    return path


def atomic_write_json(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    return atomic_write_text(path=path, text=payload)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, mode="rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_required_files(root_dir, required_files, expected_files=None, verify_hashes=True):
    root_dir = os.path.abspath(str(root_dir))
    expected_files = {} if expected_files is None else dict(expected_files)
    file_records = {}
    for relative_path in required_files:
        relative_path = str(relative_path).replace("\\", "/")
        if relative_path.startswith("/") or ".." in relative_path.split("/"):
            raise ValueError("Unsafe resource-relative path: {}".format(relative_path))
        path = os.path.join(root_dir, *relative_path.split("/"))
        if os.path.islink(path) or (not os.path.isfile(path)):
            raise FileNotFoundError("Required resource file was not found: {}".format(path))
        size = int(os.path.getsize(path))
        expected = expected_files.get(relative_path, {})
        if ("size" in expected) and size != int(expected["size"]):
            raise ValueError(
                "Resource file size mismatch for {}: expected {}, got {}".format(
                    relative_path, expected["size"], size
                )
            )
        checksum = sha256_file(path)
        if verify_hashes and expected.get("sha256") and checksum != str(expected["sha256"]).lower():
            raise ValueError("Resource SHA-256 mismatch for {}".format(relative_path))
        file_records[relative_path] = {"size": size, "sha256": checksum}
    return file_records


def is_directory_resource_ready(resource_dir, resource_id, verify_hashes=False):
    resource_dir = os.path.abspath(str(resource_dir))
    if os.path.islink(resource_dir):
        return False
    manifest_path = os.path.join(resource_dir, RESOURCE_MANIFEST_NAME)
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(manifest, dict):
        return False
    if manifest.get("format") != MANIFEST_FORMAT or manifest.get("resource_id") != str(resource_id):
        return False
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) == 0:
        return False
    for relative_path, record in files.items():
        if not isinstance(record, dict):
            return False
        relative_path = str(relative_path).replace("\\", "/")
        if relative_path.startswith("/") or ".." in relative_path.split("/"):
            return False
        path = os.path.join(resource_dir, *relative_path.split("/"))
        if os.path.islink(path) or (not os.path.isfile(path)):
            return False
        try:
            if int(os.path.getsize(path)) != int(record.get("size", -1)):
                return False
        except (TypeError, ValueError):
            return False
        if verify_hashes and sha256_file(path) != str(record.get("sha256", "")).lower():
            return False
    return True


def ensure_directory_resource(
    resource_id,
    resource_dir,
    populate,
    required_files,
    manifest_metadata=None,
    expected_files=None,
    cache_dir=None,
    no_download=False,
    verify_existing=False,
    poll_seconds=DEFAULT_LOCK_POLL_SECONDS,
    timeout_seconds=DEFAULT_LOCK_TIMEOUT_SECONDS,
):
    resource_id = str(resource_id)
    poll_seconds = float(poll_seconds)
    timeout_seconds = float(timeout_seconds)
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be > 0.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0.")
    resource_dir = os.path.abspath(os.path.expanduser(str(resource_dir)))
    managed_cache_dir = resolve_cache_dir(cache_dir)
    try:
        is_managed_path = os.path.commonpath([resource_dir, managed_cache_dir]) == managed_cache_dir
    except ValueError:
        is_managed_path = False
    if not is_managed_path:
        raise ValueError("Managed resource path should be inside the CSUBST cache directory: {}".format(resource_dir))
    if is_directory_resource_ready(resource_dir, resource_id, verify_hashes=verify_existing):
        return resource_dir
    if no_download:
        raise FileNotFoundError(
            "Required resource is not available locally and automatic download is disabled: {}".format(resource_dir)
        )
    lock_path = resolve_resource_lock_path(resource_id=resource_id, cache_dir=cache_dir)
    with acquire_exclusive_lock(
        lock_path=lock_path,
        lock_label="{} download".format(resource_id),
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    ):
        if is_directory_resource_ready(resource_dir, resource_id, verify_hashes=verify_existing):
            return resource_dir
        parent = os.path.dirname(resource_dir)
        os.makedirs(parent, exist_ok=True)
        stage_parent = os.path.join(os.path.dirname(resource_dir), ".csubst-staging")
        os.makedirs(stage_parent, exist_ok=True)
        stage_dir = tempfile.mkdtemp(prefix="{}-".format(_sanitize_resource_name(resource_id)), dir=stage_parent)
        try:
            populate(stage_dir)
            file_records = validate_required_files(
                root_dir=stage_dir,
                required_files=required_files,
                expected_files=expected_files,
                verify_hashes=True,
            )
            manifest = {
                "format": MANIFEST_FORMAT,
                "resource_id": resource_id,
                "created_at": time.time(),
                "files": file_records,
                "metadata": {} if manifest_metadata is None else dict(manifest_metadata),
            }
            atomic_write_json(os.path.join(stage_dir, RESOURCE_MANIFEST_NAME), manifest)
            if os.path.lexists(resource_dir):
                if os.path.islink(resource_dir) or (not os.path.isdir(resource_dir)):
                    raise NotADirectoryError("Resource path exists but is not a directory: {}".format(resource_dir))
                if is_directory_resource_ready(resource_dir, resource_id, verify_hashes=verify_existing):
                    return resource_dir
                shutil.rmtree(resource_dir)
            os.replace(stage_dir, resource_dir)
            stage_dir = None
        finally:
            if stage_dir is not None:
                shutil.rmtree(stage_dir, ignore_errors=True)
    return resource_dir
