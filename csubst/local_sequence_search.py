"""Run structure similarity searches against user-supplied local protein databases."""

from collections.abc import Iterable
from functools import lru_cache
import math
import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from csubst import ncbi_blast, runtime


_UNIPROT_ACCESSION = re.compile(r'(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})')
_PDB_TARGET = re.compile(r'(pdb_[0-9]{5}[a-z0-9]{3}|[0-9][a-z0-9]{3})(?:[_.][a-z0-9]+)?', re.IGNORECASE)
_MAX_CACHED_QUERIES = 32


class LocalSearchCache:
    """Bound local hit reuse to one analysis, including successful empty results."""

    def __init__(self) -> None:
        self.blastp_accessions = lru_cache(maxsize=_MAX_CACHED_QUERIES)(blastp_accessions)
        self.mmseqs_pdb_ids = lru_cache(maxsize=_MAX_CACHED_QUERIES)(mmseqs_pdb_ids)


def _query_fasta(sequence: str) -> str:
    if not re.fullmatch(r'[A-Za-z*]+', sequence):
        raise ValueError('The local search query must be a nonempty amino-acid sequence.')
    return '>query\n{}\n'.format(sequence)


def _run(
    command: list[str], *, input_text: str | None = None, env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command, input=input_text, env=env, check=True, text=True, encoding='utf-8', errors='replace',
            stdout=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        raise ValueError('Local sequence search requires {} on PATH.'.format(command[0])) from None
    except OSError as exc:
        raise ValueError('Could not start local {} ({}).'.format(command[0], exc.strerror)) from None
    except subprocess.CalledProcessError as exc:
        # Tool diagnostics can include the query. Report the failure without echoing them.
        raise ValueError(
            'Local {} failed (exit status {}). Check that the configured protein database is readable and compatible.'
            .format(command[0], exc.returncode)
        ) from None


def blastp_accessions(sequence: str, database: str, evalue: float, threads: int) -> list[str]:
    print('Running local BLAST+ against the UniProt sequence database.')
    if '"' in database:
        raise ValueError('--blastp_db cannot contain a double quote.')
    env = dict(os.environ, BLAST_USAGE_REPORT='false', NCBI_CONFIG__BLAST__DATA_LOADERS='none')
    # BLAST interprets the argument as a list of database names.
    command = [
        'blastp', '-db', '"{}"'.format(database), '-query', '-', '-outfmt', '16',
        '-evalue', str(evalue), '-max_target_seqs', str(ncbi_blast.DEFAULT_HITLIST_SIZE),
        '-num_threads', str(threads),
    ]
    result = _run(command, input_text=_query_fasta(sequence), env=env)
    hits = ncbi_blast.parse_xml2_hits(result.stdout)
    accessions = [hit.accession.upper() for hit in hits]
    if any(_UNIPROT_ACCESSION.fullmatch(accession) is None for accession in accessions):
        raise ValueError('The local BLAST+ database must preserve UniProt accessions. Check --blastp_db.')
    return accessions


def _pdb_id(identifier: str) -> str | None:
    tokens = identifier.split(maxsplit=1)
    if not tokens:
        return None
    target = tokens[0]
    pipe_id = re.fullmatch(r'pdb\|([^|]+)\|[^|]*', target, re.IGNORECASE)
    if pipe_id is not None:
        target = pipe_id[1]
    match = _PDB_TARGET.fullmatch(target)
    return match[1].lower() if match is not None else None


def _rank_pdb_hits(lines: Iterable[str]) -> list[str]:
    best_scores: dict[str, tuple[float, float]] = {}
    for line in lines:
        try:
            target, evalue_text, bits_text, header = line.rstrip('\r\n').split('\t', 3)
            evalue, bits = float(evalue_text), float(bits_text)
        except ValueError:
            raise ValueError('Local MMseqs2 produced an invalid hit table.') from None
        if not (math.isfinite(evalue) and evalue >= 0 and math.isfinite(bits)):
            raise ValueError('Local MMseqs2 produced invalid hit scores.')
        pdb_id = _pdb_id(header) or _pdb_id(target)
        if pdb_id is None:
            raise ValueError('Local MMseqs2 hits must contain PDB identifiers. Check the headers in --mmseqs_db.')
        score = (evalue, -bits)
        best_scores[pdb_id] = min(score, best_scores.get(pdb_id, score))
    return sorted(best_scores, key=lambda pdb_id: (*best_scores[pdb_id], pdb_id))


def mmseqs_pdb_ids(
    sequence: str, database: str, evalue: float, minimum_identity: float, threads: int,
) -> list[str]:
    print('Running local MMseqs2 against the PDB sequence database.')
    if not Path(database + '.dbtype').is_file():
        raise ValueError('--mmseqs_db must point to an existing MMseqs2 database prefix (missing .dbtype).')
    with TemporaryDirectory(prefix='mmseqs-', dir=runtime.get_run_tempdir()) as dirname:
        workdir = Path(dirname)
        query = workdir / 'query.fasta'
        query.write_text(_query_fasta(sequence), encoding='ascii')
        output = workdir / 'hits.tsv'
        _run([
            'mmseqs', 'easy-search', str(query), database, str(output), str(workdir / 'tmp'),
            '--search-type', '1', '--alignment-mode', '3', '-e', str(evalue),
            '--min-seq-id', str(minimum_identity), '--threads', str(threads),
            '--format-output', 'target,evalue,bits,theader',
        ])
        try:
            with output.open(encoding='utf-8') as hits:
                return _rank_pdb_hits(hits)
        except (OSError, UnicodeError):
            raise ValueError('Could not read the local MMseqs2 hit table.') from None
