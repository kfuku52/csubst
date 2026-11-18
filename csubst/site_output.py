"""Output directory names shared by CSUBST sites and its consumers."""

from hashlib import sha256


def intersection_job_dirname(branch_ids: str, output_prefix: str = 'csubst') -> str:
    """Return the directory name for an ordered, comma-separated list of integer IDs.

    Use the same UTF-8 byte limit and digest across runtimes so a consumer can
    resolve an output directory even after copying or extracting it.
    """
    legacy_name = output_prefix + '.branch_id' + branch_ids
    encoded = legacy_name.encode('utf-8')
    if len(encoded) <= 255:
        return legacy_name
    return 'csubst.branch_id.sha256-' + sha256(encoded).hexdigest()[:32]
