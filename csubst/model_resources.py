import os

from csubst import resource_cache


VESM_35M_RESOURCE_ID = (
    "vesm-35m@946749efe0c5bda4693884092df3a73c542752f3+"
    "esm2-t12-35m@6fbf070e65b0b7291e7bbcd451118c216cff79d8"
)
VESM_REPO_ID = "ntranoslab/vesm"
VESM_REVISION = "946749efe0c5bda4693884092df3a73c542752f3"
VESM_CHECKPOINT_FILENAME = "VESM_35M.pth"
VESM_BASE_REPO_ID = "facebook/esm2_t12_35M_UR50D"
VESM_BASE_REVISION = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"
VESM_BASE_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
)
VESM_REQUIRED_FILES = (VESM_CHECKPOINT_FILENAME,) + tuple(
    "base_model/{}".format(file_name) for file_name in VESM_BASE_FILES
)
VESM_EXPECTED_FILES = {
    VESM_CHECKPOINT_FILENAME: {
        "size": 136063095,
        "sha256": "0667fcbd3605a1a953e4b991ba67ac8df248a03331d617839605fd5691f2cc7d",
    },
    "base_model/model.safetensors": {
        "size": 136008798,
        "sha256": "e35647818e0e064351d4531ed480d225a002567b4b2b93ad3a9246d753150fc0",
    },
}


def _import_hf_hub_download():
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ImportError(
            "VESM-35M download requires huggingface-hub. "
            "Install it with `pip install huggingface-hub` or `pip install csubst[vep]`."
        ) from exc
    return hf_hub_download


def resolve_vesm35m_resource_dir(cache_dir=None):
    return os.path.join(resource_cache.resolve_cache_dir(cache_dir), "models", "vesm-35m", "v1")


def get_vesm35m_paths(resource_dir):
    resource_dir = os.path.abspath(str(resource_dir))
    return {
        "resource_dir": resource_dir,
        "checkpoint_path": os.path.join(resource_dir, VESM_CHECKPOINT_FILENAME),
        "base_model_dir": os.path.join(resource_dir, "base_model"),
        "manifest_path": os.path.join(resource_dir, resource_cache.RESOURCE_MANIFEST_NAME),
    }


def ensure_vesm35m_resource(
    cache_dir=None,
    no_download=False,
    verify_existing=False,
    poll_seconds=resource_cache.DEFAULT_LOCK_POLL_SECONDS,
    timeout_seconds=resource_cache.DEFAULT_LOCK_TIMEOUT_SECONDS,
    download_file=None,
):
    managed_cache_dir = resource_cache.resolve_cache_dir(cache_dir)
    resource_dir = resolve_vesm35m_resource_dir(cache_dir=managed_cache_dir)

    def populate(stage_dir):
        active_download_file = download_file
        if active_download_file is None:
            active_download_file = _import_hf_hub_download()
        base_dir = os.path.join(stage_dir, "base_model")
        os.makedirs(base_dir, exist_ok=True)
        print(
            "Downloading VESM-35M checkpoint from {}/{} at revision {}.".format(
                VESM_REPO_ID, VESM_CHECKPOINT_FILENAME, VESM_REVISION
            ),
            flush=True,
        )
        active_download_file(
            repo_id=VESM_REPO_ID,
            filename=VESM_CHECKPOINT_FILENAME,
            revision=VESM_REVISION,
            local_dir=stage_dir,
        )
        print(
            "Downloading VESM-35M base model {} at revision {}.".format(
                VESM_BASE_REPO_ID, VESM_BASE_REVISION
            ),
            flush=True,
        )
        for filename in VESM_BASE_FILES:
            active_download_file(
                repo_id=VESM_BASE_REPO_ID,
                filename=filename,
                revision=VESM_BASE_REVISION,
                local_dir=base_dir,
            )

    resource_cache.ensure_directory_resource(
        resource_id=VESM_35M_RESOURCE_ID,
        resource_dir=resource_dir,
        populate=populate,
        required_files=VESM_REQUIRED_FILES,
        manifest_metadata={
            "model": "VESM_35M",
            "checkpoint_repo_id": VESM_REPO_ID,
            "checkpoint_revision": VESM_REVISION,
            "base_model_repo_id": VESM_BASE_REPO_ID,
            "base_model_revision": VESM_BASE_REVISION,
        },
        expected_files=VESM_EXPECTED_FILES,
        cache_dir=managed_cache_dir,
        no_download=no_download,
        verify_existing=verify_existing,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    )
    return get_vesm35m_paths(resource_dir)
