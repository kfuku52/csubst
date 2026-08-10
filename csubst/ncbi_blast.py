from dataclasses import dataclass
import math
import os
import re
import time

from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException
import requests

from csubst import __version__


BLAST_URL = 'https://blast.ncbi.nlm.nih.gov/Blast.cgi'
DEFAULT_HITLIST_SIZE = 50
DEFAULT_POLL_INTERVAL = 60.0
DEFAULT_MAX_WAIT_SECONDS = 3600.0
MAX_RESPONSE_BYTES = 20_000_000
DEFAULT_CONTACT_EMAIL = 'kfuku52@gmail.com'


@dataclass(frozen=True)
class BlastHit:
    accession: str
    title: str


def _parse_submission_response(response_text):
    rid_match = re.search(r'^\s*RID\s*=\s*(\S+)\s*$', response_text, flags=re.MULTILINE)
    if rid_match is None:
        raise RuntimeError('NCBI BLAST submission response did not contain an RID.')
    rtoe_match = re.search(r'^\s*RTOE\s*=\s*([0-9]+)\s*$', response_text, flags=re.MULTILINE)
    rtoe = int(rtoe_match.group(1)) if rtoe_match is not None else 0
    return rid_match.group(1), rtoe


def _parse_status(response_text):
    status_match = re.search(r'^\s*Status\s*=\s*([A-Za-z]+)\s*$', response_text, flags=re.MULTILINE)
    if status_match is None:
        return None
    return status_match.group(1).upper()


def _local_name(tag):
    return str(tag).rsplit('}', 1)[-1]


def _first_descendant_text(element, local_name):
    for descendant in element.iter():
        if _local_name(descendant.tag) != local_name:
            continue
        text = str(descendant.text or '').strip()
        if text != '':
            return text
    return ''


def _accession_from_identifier(identifier):
    pipe_match = re.search(r'\|([^|]+)\|', identifier)
    if pipe_match is not None:
        accession = pipe_match.group(1)
    else:
        tokens = identifier.split()
        accession = tokens[0] if len(tokens) else ''
    return re.sub(r'\..*', '', accession).strip()


def parse_xml2_hits(response_text):
    """Extract ordered primary hit accessions and display titles from BLAST XML2."""

    if len(response_text.encode('utf-8')) > MAX_RESPONSE_BYTES:
        raise ValueError('NCBI BLAST XML2 output exceeded the response size limit.')
    try:
        root = ET.fromstring(response_text)
    except (ET.ParseError, DefusedXmlException) as exc:
        raise ValueError('NCBI BLAST returned malformed XML2 output.') from exc

    hits = list()
    seen_accessions = set()
    for hit_element in root.iter():
        if _local_name(hit_element.tag) != 'Hit':
            continue
        description_element = None
        for descendant in hit_element.iter():
            if _local_name(descendant.tag) == 'HitDescr':
                description_element = descendant
                break
        if description_element is None:
            description_element = hit_element
        identifier = _first_descendant_text(description_element, 'id')
        accession = _first_descendant_text(description_element, 'accession')
        title = _first_descendant_text(description_element, 'title')
        if accession == '':
            accession = _accession_from_identifier(identifier or title)
        else:
            accession = re.sub(r'\..*', '', accession).strip()
        if (accession == '') or (accession in seen_accessions):
            continue
        seen_accessions.add(accession)
        display_title = ' '.join([token for token in [identifier, title] if token != ''])
        if display_title == '':
            display_title = accession
        hits.append(BlastHit(accession=accession, title=display_title))
    return hits


def _post_text(session, data, timeout, url):
    response = session.post(
        url,
        data=data,
        headers={'User-Agent': 'csubst/{}'.format(__version__)},
        timeout=timeout,
        stream=True,
    )
    try:
        response.raise_for_status()
        content_length = response.headers.get('Content-Length')
        if content_length is not None and int(content_length) > MAX_RESPONSE_BYTES:
            raise RuntimeError('NCBI BLAST response exceeded the response size limit.')
        payload = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            payload.extend(chunk)
            if len(payload) > MAX_RESPONSE_BYTES:
                raise RuntimeError('NCBI BLAST response exceeded the response size limit.')
        return payload.decode(response.encoding or 'utf-8', errors='replace')
    finally:
        response.close()


def _sleep_until_next_poll(seconds, *, deadline, monotonic, sleep, rid):
    remaining = deadline - monotonic()
    if remaining <= 0 or seconds > remaining:
        raise TimeoutError(
            'NCBI BLAST search {} exceeded the overall wait limit.'.format(rid)
        )
    sleep(seconds)


def _remaining_request_timeout(timeout, *, deadline, monotonic, rid):
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError(
            'NCBI BLAST search {} exceeded the overall wait limit.'.format(rid)
        )
    return min(float(timeout), remaining)


def search_blastp_swissprot(
    sequence,
    expect=10,
    timeout=30,
    hitlist_size=DEFAULT_HITLIST_SIZE,
    poll_interval=DEFAULT_POLL_INTERVAL,
    max_wait=DEFAULT_MAX_WAIT_SECONDS,
    email=None,
    session=None,
    sleep=time.sleep,
    monotonic=time.monotonic,
    url=BLAST_URL,
):
    """Submit and retrieve one blastp search against NCBI Swiss-Prot."""

    sequence = str(sequence).strip()
    if sequence == '':
        raise ValueError('NCBI BLAST query sequence should not be empty.')
    timeout = float(timeout)
    if (not math.isfinite(timeout)) or timeout <= 0:
        raise ValueError('NCBI BLAST request timeout should be finite and > 0.')
    expect = float(expect)
    if (not math.isfinite(expect)) or expect <= 0:
        raise ValueError('NCBI BLAST expect value should be finite and > 0.')
    hitlist_size = int(hitlist_size)
    if hitlist_size <= 0:
        raise ValueError('NCBI BLAST hitlist_size should be > 0.')
    poll_interval = float(poll_interval)
    if (not math.isfinite(poll_interval)) or poll_interval < DEFAULT_POLL_INTERVAL:
        raise ValueError('NCBI BLAST poll_interval should be finite and at least 60 seconds.')
    max_wait = float(max_wait)
    if (not math.isfinite(max_wait)) or max_wait <= 0:
        raise ValueError('NCBI BLAST max_wait should be finite and > 0.')
    deadline = monotonic() + max_wait

    owns_session = session is None
    if session is None:
        session = requests.Session()
    tool_name = 'csubst-{}'.format(__version__)
    contact_email = str(
        email or os.environ.get('NCBI_EMAIL', '') or DEFAULT_CONTACT_EMAIL
    ).strip()
    common_parameters = {'tool': tool_name, 'email': contact_email}

    try:
        submission_parameters = {
            'CMD': 'Put',
            'PROGRAM': 'blastp',
            'DATABASE': 'swissprot',
            'QUERY': sequence,
            'EXPECT': expect,
            'HITLIST_SIZE': hitlist_size,
            'FORMAT_TYPE': 'XML2_S',
        }
        submission_parameters.update(common_parameters)
        submission_text = _post_text(
            session,
            submission_parameters,
            _remaining_request_timeout(
                timeout, deadline=deadline, monotonic=monotonic, rid='submission'
            ),
            url,
        )
        rid, rtoe = _parse_submission_response(submission_text)

        retrieval_parameters = {
            'CMD': 'Get',
            'RID': rid,
            'FORMAT_TYPE': 'XML2_S',
        }
        retrieval_parameters.update(common_parameters)
        _sleep_until_next_poll(
            max(poll_interval, float(rtoe)),
            deadline=deadline,
            monotonic=monotonic,
            sleep=sleep,
            rid=rid,
        )
        ready_status_seen = False
        while True:
            if monotonic() >= deadline:
                raise TimeoutError(
                    'NCBI BLAST search {} exceeded the overall wait limit.'.format(rid)
                )
            result_text = _post_text(
                session,
                retrieval_parameters,
                _remaining_request_timeout(
                    timeout, deadline=deadline, monotonic=monotonic, rid=rid
                ),
                url,
            )
            status = _parse_status(result_text)
            if status is None:
                return parse_xml2_hits(result_text)
            if status == 'WAITING':
                _sleep_until_next_poll(
                    poll_interval,
                    deadline=deadline,
                    monotonic=monotonic,
                    sleep=sleep,
                    rid=rid,
                )
                continue
            if status in {'FAILED', 'UNKNOWN'}:
                raise RuntimeError('NCBI BLAST search {} returned status {}.'.format(rid, status))
            if status == 'READY':
                if ready_status_seen:
                    raise RuntimeError('NCBI BLAST search {} repeatedly returned READY without results.'.format(rid))
                ready_status_seen = True
                _sleep_until_next_poll(
                    poll_interval,
                    deadline=deadline,
                    monotonic=monotonic,
                    sleep=sleep,
                    rid=rid,
                )
                continue
            raise RuntimeError('NCBI BLAST search {} returned unsupported status {}.'.format(rid, status))
    finally:
        if owns_session:
            session.close()
