import hashlib
import os
import re
import urllib.request

from csubst import resource_cache


def _sanitize_component(value, label):
    raw = str(value).strip()
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    if normalized == "":
        raise ValueError("{} should contain at least one safe file-name character.".format(label))
    return normalized


def _validate_filename(filename):
    filename = str(filename).strip()
    if (filename == "") or (filename != os.path.basename(filename)) or filename in [".", ".."]:
        raise ValueError("Structure cache filename should be a file name, not a path: {}".format(filename))
    return filename


def _download_url_bytes(url, timeout_seconds):
    with urllib.request.urlopen(str(url), timeout=float(timeout_seconds)) as response:
        return response.read()


def ensure_remote_structure(
    source,
    structure_id,
    url,
    filename,
    cache_dir=None,
    network_timeout=30.0,
    poll_seconds=resource_cache.DEFAULT_LOCK_POLL_SECONDS,
    lock_timeout_seconds=resource_cache.DEFAULT_LOCK_TIMEOUT_SECONDS,
    download_bytes=None,
):
    source_name = _sanitize_component(source, label="Structure source")
    structure_name = _sanitize_component(structure_id, label="Structure ID")
    filename = _validate_filename(filename)
    url = str(url).strip()
    if url == "":
        raise ValueError("Structure download URL should be non-empty.")
    url_sha256 = hashlib.sha256(url.encode("utf-8")).hexdigest()
    managed_cache_dir = resource_cache.resolve_cache_dir(cache_dir)
    resource_dir = os.path.join(
        managed_cache_dir,
        "structures",
        source_name.lower(),
        structure_name.lower(),
        url_sha256[:16],
    )
    resource_id = "structure-{}-{}@{}".format(
        source_name.lower(),
        structure_name.lower(),
        url_sha256,
    )

    def populate(stage_dir):
        active_download = _download_url_bytes if download_bytes is None else download_bytes
        payload = active_download(url, float(network_timeout))
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("Structure downloader should return bytes: {}".format(url))
        if len(payload) == 0:
            raise ValueError("Downloaded structure file was empty: {}".format(url))
        with open(os.path.join(stage_dir, filename), mode="wb") as handle:
            handle.write(payload)

    resource_cache.ensure_directory_resource(
        resource_id=resource_id,
        resource_dir=resource_dir,
        populate=populate,
        required_files=[filename],
        manifest_metadata={
            "resource_type": "protein_structure",
            "source": str(source),
            "structure_id": str(structure_id),
            "url": url,
        },
        cache_dir=managed_cache_dir,
        poll_seconds=float(poll_seconds),
        timeout_seconds=float(lock_timeout_seconds),
    )
    return os.path.join(resource_dir, filename)


def ensure_rcsb_structure(
    pdb_id,
    cache_dir=None,
    network_timeout=30.0,
    poll_seconds=resource_cache.DEFAULT_LOCK_POLL_SECONDS,
    lock_timeout_seconds=resource_cache.DEFAULT_LOCK_TIMEOUT_SECONDS,
    download_bytes=None,
):
    pdb_id = str(pdb_id).strip()
    is_old_pdb_code = bool(re.fullmatch(r"[0-9][A-Za-z0-9]{3}", pdb_id))
    is_new_pdb_code = bool(re.fullmatch(r"pdb_[0-9]{5}[A-Za-z0-9]{3}", pdb_id, flags=re.IGNORECASE))
    if not (is_old_pdb_code or is_new_pdb_code):
        raise ValueError("Invalid RCSB PDB identifier: {}".format(pdb_id))
    filename = pdb_id.lower() + ".cif"
    url = "https://files.rcsb.org/download/{}.cif".format(pdb_id.upper())
    return ensure_remote_structure(
        source="rcsb",
        structure_id=pdb_id,
        url=url,
        filename=filename,
        cache_dir=cache_dir,
        network_timeout=network_timeout,
        poll_seconds=poll_seconds,
        lock_timeout_seconds=lock_timeout_seconds,
        download_bytes=download_bytes,
    )
