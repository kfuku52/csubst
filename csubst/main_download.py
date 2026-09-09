from csubst import model_resources
from csubst import structural_alphabet
from csubst import structural_prediction
from csubst.config_types import AnalysisConfig


def _normalize_resources(value: object) -> list[str]:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized == "all":
        return ["vesm-35m", "prostt5", "prostt5-cnn", "esm3di-35m"]
    if normalized in ["vesm-35m", "prostt5", "prostt5-cnn", "esm3di-35m"]:
        return [normalized]
    raise ValueError("--resource should be one of vesm-35m, prostt5, prostt5-cnn, esm3di-35m, all.")


def main_download(g: AnalysisConfig) -> None:
    resources = _normalize_resources(g.get("resource", "vesm-35m"))
    legacy_verify = g.get("verify")
    if legacy_verify and any(name in resources for name in ["prostt5", "prostt5-cnn"]):
        raise ValueError(
            "--verify yes is not supported for ProstT5: CSUBST does not perform "
            "SHA-256 verification of that resource. Use --no_download yes to "
            "check local loading only, without requesting a checksum check."
        )
    if legacy_verify is not None:
        print(
            "Warning: --verify is deprecated. VESM-35M, ESM3Di and CNN files are always "
            "SHA-256 verified, including with --verify no. Use --no_download yes "
            "to check existing resources without downloading.",
            flush=True,
        )
    poll_seconds = float(g.get("resource_lock_poll", 5.0))
    timeout_seconds = float(g.get("resource_lock_timeout", 3600.0))
    if poll_seconds <= 0:
        raise ValueError("--resource_lock_poll should be > 0.")
    if timeout_seconds <= 0:
        raise ValueError("--resource_lock_timeout should be > 0.")
    if "vesm-35m" in resources:
        try:
            paths = model_resources.ensure_vesm35m_resource(
                cache_dir=g.get("resource_cache_dir", ""),
                no_download=bool(g.get("no_download", False)),
                verify_existing=True,
                poll_seconds=poll_seconds,
                timeout_seconds=timeout_seconds,
            )
        except (FileNotFoundError, ImportError) as exc:
            raise ValueError(str(exc)) from exc
        print("VESM-35M model files are ready: {}".format(paths["resource_dir"]), flush=True)
    if "prostt5" in resources:
        prostt5_g = dict(g)
        prostt5_g["prostt5_no_download"] = bool(g.get("no_download", False))
        try:
            model_source = structural_alphabet.ensure_prostt5_model_files(g=prostt5_g)
        except (RuntimeError, ImportError) as exc:
            raise ValueError(str(exc)) from exc
        print("ProstT5 model files are ready: {}".format(model_source), flush=True)
    for backend in ["prostt5-cnn", "esm3di-35m"]:
        if backend not in resources:
            continue
        local_g = dict(g, sa_backend=backend, prostt5_no_download=bool(g.get("no_download", False)))
        try:
            model_source = structural_prediction.ensure_encoder_model_files(local_g)
        except (FileNotFoundError, RuntimeError, ImportError) as exc:
            raise ValueError(str(exc)) from exc
        print("{} model files are ready: {}".format(backend, model_source), flush=True)
