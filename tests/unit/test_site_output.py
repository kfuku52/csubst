from hashlib import sha256

from csubst.site_output import intersection_job_dirname


def test_intersection_job_name_uses_utf8_byte_limit():
    prefix = 'é' * 121
    short = prefix + '.branch_id1,2'
    assert len(short.encode('utf-8')) == 255
    assert intersection_job_dirname('1,2', prefix) == short

    long = prefix + '.branch_id1,23'
    digest = sha256(long.encode('utf-8')).hexdigest()[:32]
    assert intersection_job_dirname('1,23', prefix) == 'csubst.branch_id.sha256-' + digest


def test_shortened_intersection_name_preserves_branch_order():
    branches = [str(branch_id) for branch_id in range(100, 181)]
    forward = intersection_job_dirname(','.join(branches))
    reverse = intersection_job_dirname(','.join(reversed(branches)))
    assert forward.startswith('csubst.branch_id.sha256-')
    assert reverse.startswith('csubst.branch_id.sha256-')
    assert forward != reverse
