"""Importable process targets for both source and installed-wheel tests.

Spawned interpreters cannot import pytest importlib-mode synthetic modules.
Keep targets here, on the shared support path, without exposing the checkout.
"""

import os


def _concurrent_resource_worker(
    cache_dir,
    start_event,
    populate_started_event,
    release_populate_event,
    result_queue,
):
    from csubst import resource_cache

    resource_dir = os.path.join(cache_dir, "models", "demo", "v1")
    counter_path = os.path.join(cache_dir, "populate-count.txt")

    def populate(stage_dir):
        with open(counter_path, mode="a", encoding="utf-8") as handle:
            handle.write("populate\n")
            handle.flush()
            os.fsync(handle.fileno())
        populate_started_event.set()
        if not release_populate_event.wait(timeout=5):
            raise TimeoutError("Timed out waiting to release test resource population.")
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
    from csubst import structural_alphabet

    start_event.wait(timeout=5)
    structural_alphabet._append_prostt5_sequence_cache(
        cache_file=cache_path,
        model_key="demo-model",
        seq_to_pred={sequence: "A" * len(sequence)},
        poll_seconds=0.01,
        timeout_seconds=5,
    )


def _starmap_add_mul(a, b):
    return (a + b) * 2
