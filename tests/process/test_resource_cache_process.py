"""Cross-process cache ownership and concurrent cache updates."""

import multiprocessing

import pytest

from process_workers import _concurrent_resource_worker, _sequence_cache_worker


@pytest.mark.slow
@pytest.mark.process
def test_independent_processes_populate_shared_resource_once(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    start_event = ctx.Event()
    populate_started_event = ctx.Event()
    release_populate_event = ctx.Event()
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_concurrent_resource_worker,
            args=(
                str(tmp_path / "cache"),
                start_event,
                populate_started_event,
                release_populate_event,
                result_queue,
            ),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start_event.set()
    try:
        assert populate_started_event.wait(timeout=10)
    finally:
        release_populate_event.set()
    results = [result_queue.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    assert [status for status, _ in results] == ["ok", "ok"]
    counter_path = tmp_path / "cache" / "populate-count.txt"
    assert counter_path.read_text(encoding="utf-8").splitlines() == ["populate"]


@pytest.mark.slow
@pytest.mark.process
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
