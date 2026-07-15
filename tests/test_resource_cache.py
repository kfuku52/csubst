import json
import multiprocessing
import os
import socket
import time

import pytest

from csubst import model_resources
from csubst import resource_cache
from csubst import structural_alphabet


def _concurrent_resource_worker(cache_dir, start_event, result_queue):
    resource_dir = os.path.join(cache_dir, "models", "demo", "v1")
    counter_path = os.path.join(cache_dir, "populate-count.txt")

    def populate(stage_dir):
        with open(counter_path, mode="a", encoding="utf-8") as handle:
            handle.write("populate\n")
            handle.flush()
            os.fsync(handle.fileno())
        time.sleep(0.25)
        with open(os.path.join(stage_dir, "payload.txt"), mode="w", encoding="utf-8") as handle:
            handle.write("ready\n")

    start_event.wait(timeout=5)
    try:
        out = resource_cache.ensure_directory_resource(
            resource_id="demo-resource-v1",
            resource_dir=resource_dir,
            populate=populate,
            required_files=["payload.txt"],
            cache_dir=cache_dir,
            poll_seconds=0.02,
            timeout_seconds=10,
        )
        result_queue.put(("ok", out))
    except Exception as exc:  # pragma: no cover - surfaced in the parent process
        result_queue.put(("error", repr(exc)))


def _sequence_cache_worker(cache_path, start_event, sequence):
    start_event.wait(timeout=5)
    structural_alphabet._append_prostt5_sequence_cache(
        cache_file=cache_path,
        model_key="demo-model",
        seq_to_pred={sequence: "A" * len(sequence)},
        poll_seconds=0.01,
        timeout_seconds=5,
    )


def test_resolve_cache_dir_prefers_explicit_path(tmp_path, monkeypatch):
    monkeypatch.setenv("CSUBST_CACHE_DIR", str(tmp_path / "env-cache"))
    assert resource_cache.resolve_cache_dir(tmp_path / "explicit") == str(tmp_path / "explicit")


def test_acquire_exclusive_lock_rejects_symlink(tmp_path):
    lock_path = tmp_path / "resource.lock"
    os.symlink(tmp_path / "missing-target", lock_path)
    with pytest.raises(IsADirectoryError, match="lock path"):
        with resource_cache.acquire_exclusive_lock(
            lock_path,
            heartbeat_seconds=0.05,
            stale_seconds=0.2,
            poll_seconds=0.01,
            timeout_seconds=0.1,
        ):
            pass


def test_acquire_exclusive_lock_reclaims_dead_same_host_owner(tmp_path, monkeypatch):
    lock_path = tmp_path / "resource.lock"
    lock_path.write_text(
        json.dumps(
            {
                "format": resource_cache.LOCK_FORMAT,
                "owner_token": "dead-owner",
                "hostname": socket.gethostname(),
                "boot_id": "boot-test",
                "pid": 999999,
                "created_at": time.time(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(resource_cache, "_resolve_local_boot_id", lambda: "boot-test")
    monkeypatch.setattr(resource_cache, "_is_process_alive", lambda pid: False)
    with resource_cache.acquire_exclusive_lock(
        lock_path,
        heartbeat_seconds=0.05,
        stale_seconds=0.2,
        poll_seconds=0.01,
        timeout_seconds=1,
    ):
        metadata = json.loads(lock_path.read_text(encoding="utf-8"))
        assert metadata["pid"] == os.getpid()
        assert metadata["owner_token"] != "dead-owner"
    assert not lock_path.exists()


def test_acquire_exclusive_lock_heartbeat_keeps_lock_fresh(tmp_path):
    lock_path = tmp_path / "resource.lock"
    with resource_cache.acquire_exclusive_lock(
        lock_path,
        heartbeat_seconds=0.02,
        stale_seconds=0.2,
        poll_seconds=0.01,
        timeout_seconds=1,
    ):
        before = lock_path.stat().st_mtime_ns
        time.sleep(0.08)
        after = lock_path.stat().st_mtime_ns
        assert after > before


def test_acquire_exclusive_lock_does_not_remove_successor_lock(tmp_path):
    lock_path = tmp_path / "resource.lock"
    successor = {
        "format": resource_cache.LOCK_FORMAT,
        "owner_token": "successor-owner",
        "hostname": "remote-node",
        "boot_id": "remote-boot",
        "pid": 123,
        "created_at": time.time(),
    }
    with resource_cache.acquire_exclusive_lock(
        lock_path,
        heartbeat_seconds=0.05,
        stale_seconds=0.2,
        poll_seconds=0.01,
        timeout_seconds=1,
    ):
        lock_path.unlink()
        lock_path.write_text(json.dumps(successor) + "\n", encoding="utf-8")
    assert json.loads(lock_path.read_text(encoding="utf-8"))["owner_token"] == "successor-owner"


def test_acquire_exclusive_lock_times_out_for_live_same_host_owner(tmp_path, monkeypatch):
    lock_path = tmp_path / "resource.lock"
    lock_path.write_text(
        json.dumps(
            {
                "format": resource_cache.LOCK_FORMAT,
                "owner_token": "live-owner",
                "hostname": socket.gethostname(),
                "boot_id": "boot-test",
                "pid": 12345,
                "created_at": time.time(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(resource_cache, "_resolve_local_boot_id", lambda: "boot-test")
    monkeypatch.setattr(resource_cache, "_is_process_alive", lambda pid: True)
    with pytest.raises(TimeoutError, match="live-owner|Timed out"):
        with resource_cache.acquire_exclusive_lock(
            lock_path,
            heartbeat_seconds=0.02,
            stale_seconds=0.1,
            poll_seconds=0.01,
            timeout_seconds=0.05,
        ):
            pass
    assert lock_path.exists()


def test_directory_resource_is_atomically_published_with_manifest(tmp_path):
    cache_dir = tmp_path / "cache"
    resource_dir = cache_dir / "models" / "demo" / "v1"
    calls = []

    def populate(stage_dir):
        calls.append(stage_dir)
        with open(os.path.join(stage_dir, "payload.txt"), mode="w", encoding="utf-8") as handle:
            handle.write("payload")

    out = resource_cache.ensure_directory_resource(
        resource_id="demo-v1",
        resource_dir=resource_dir,
        populate=populate,
        required_files=["payload.txt"],
        cache_dir=cache_dir,
        poll_seconds=0.01,
        timeout_seconds=1,
    )
    assert out == str(resource_dir)
    assert len(calls) == 1
    assert resource_cache.is_directory_resource_ready(resource_dir, "demo-v1", verify_hashes=True)
    manifest = json.loads((resource_dir / resource_cache.RESOURCE_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["files"]["payload.txt"]["size"] == len("payload")

    resource_cache.ensure_directory_resource(
        resource_id="demo-v1",
        resource_dir=resource_dir,
        populate=lambda stage_dir: (_ for _ in ()).throw(AssertionError("should not repopulate")),
        required_files=["payload.txt"],
        cache_dir=cache_dir,
        poll_seconds=0.01,
        timeout_seconds=1,
    )
    assert len(calls) == 1


def test_failed_population_leaves_no_partial_resource_and_can_retry(tmp_path):
    cache_dir = tmp_path / "cache"
    resource_dir = cache_dir / "models" / "demo" / "v1"

    def fail_after_partial_write(stage_dir):
        with open(os.path.join(stage_dir, "payload.txt"), mode="w", encoding="utf-8") as handle:
            handle.write("partial")
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        resource_cache.ensure_directory_resource(
            resource_id="demo-v1",
            resource_dir=resource_dir,
            populate=fail_after_partial_write,
            required_files=["payload.txt"],
            cache_dir=cache_dir,
            poll_seconds=0.01,
            timeout_seconds=1,
        )
    assert not resource_dir.exists()

    def populate(stage_dir):
        with open(os.path.join(stage_dir, "payload.txt"), mode="w", encoding="utf-8") as handle:
            handle.write("complete")

    resource_cache.ensure_directory_resource(
        resource_id="demo-v1",
        resource_dir=resource_dir,
        populate=populate,
        required_files=["payload.txt"],
        cache_dir=cache_dir,
        poll_seconds=0.01,
        timeout_seconds=1,
    )
    assert (resource_dir / "payload.txt").read_text(encoding="utf-8") == "complete"


def test_independent_processes_populate_shared_resource_once(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    start_event = ctx.Event()
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_concurrent_resource_worker,
            args=(str(tmp_path / "cache"), start_event, result_queue),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start_event.set()
    results = [result_queue.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    assert [status for status, _ in results] == ["ok", "ok"]
    counter_path = tmp_path / "cache" / "populate-count.txt"
    assert counter_path.read_text(encoding="utf-8").splitlines() == ["populate"]


def test_independent_processes_merge_prostt5_sequence_cache_updates(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    start_event = ctx.Event()
    cache_path = str(tmp_path / "prostt5.tsv")
    processes = [
        ctx.Process(target=_sequence_cache_worker, args=(cache_path, start_event, sequence))
        for sequence in ["AC", "MNP"]
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    lines = (tmp_path / "prostt5.tsv").read_text(encoding="utf-8").splitlines()
    assert sorted(lines) == sorted(["demo-model\tAC\tAA", "demo-model\tMNP\tAAA"])


def test_ensure_vesm35m_downloads_only_required_pinned_files(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(model_resources, "VESM_EXPECTED_FILES", {})

    def fake_download_file(repo_id, filename, revision, local_dir):
        calls.append((repo_id, filename, revision))
        os.makedirs(local_dir, exist_ok=True)
        path = os.path.join(local_dir, filename)
        with open(path, mode="wb") as handle:
            handle.write((repo_id + ":" + filename + ":" + revision).encode("utf-8"))
        return path

    paths = model_resources.ensure_vesm35m_resource(
        cache_dir=tmp_path / "cache",
        poll_seconds=0.01,
        timeout_seconds=1,
        download_file=fake_download_file,
    )
    assert os.path.isfile(paths["checkpoint_path"])
    assert os.path.isfile(os.path.join(paths["base_model_dir"], "model.safetensors"))
    assert len(calls) == 1 + len(model_resources.VESM_BASE_FILES)
    assert calls[0] == (
        model_resources.VESM_REPO_ID,
        model_resources.VESM_CHECKPOINT_FILENAME,
        model_resources.VESM_REVISION,
    )
    calls.clear()
    model_resources.ensure_vesm35m_resource(
        cache_dir=tmp_path / "cache",
        poll_seconds=0.01,
        timeout_seconds=1,
        download_file=fake_download_file,
    )
    assert calls == []
