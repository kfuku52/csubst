from csubst import model_resources
from csubst import structural_alphabet


def _normalize_resources(value):
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized == "all":
        return ["vesm-35m", "prostt5"]
    if normalized in ["vesm-35m", "prostt5"]:
        return [normalized]
    raise ValueError("--resource should be one of vesm-35m, prostt5, all.")


def main_download(g):
    resources = _normalize_resources(g.get("resource", "vesm-35m"))
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
                verify_existing=bool(g.get("verify", False)),
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
