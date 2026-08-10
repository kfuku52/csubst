import gzip
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class FastaRecord:
    """A minimal FASTA record with explicit header and identifier semantics."""

    description: str
    sequence: str

    @property
    def id(self) -> str:
        return self.description.split(None, 1)[0]

    @property
    def name(self) -> str:
        return self.id


def _open_text_source(source: Any) -> Any:
    if hasattr(source, 'read'):
        return nullcontext(source)
    path = str(source)
    if path.lower().endswith('.gz'):
        return gzip.open(path, mode='rt', encoding='utf-8')
    return open(path, mode='rt', encoding='utf-8')


def _open_text_destination(destination: Any) -> Any:
    if hasattr(destination, 'write'):
        return nullcontext(destination)
    path = str(destination)
    if path.lower().endswith('.gz'):
        return gzip.open(path, mode='wt', encoding='utf-8', newline='\n')
    return open(path, mode='w', encoding='utf-8', newline='\n')


def _source_label(source: Any) -> str:
    if isinstance(source, (str, Path)):
        return str(source)
    return str(getattr(source, 'name', '<stream>'))


def read_fasta_records(source: Any) -> list[FastaRecord]:
    """Read FASTA records from a path or text stream, preserving input order."""

    records: list[FastaRecord] = []
    description: str | None = None
    sequence_parts: list[str] = []
    source_label = _source_label(source)
    with _open_text_source(source) as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip('\r\n')
            if line.startswith('>'):
                if description is not None:
                    records.append(FastaRecord(description, ''.join(sequence_parts)))
                description = line[1:].strip()
                if description == '':
                    txt = 'Invalid FASTA header in {} at line {}: sequence name is empty.'
                    raise ValueError(txt.format(source_label, line_no))
                sequence_parts = list()
                continue
            if line.strip() == '':
                continue
            if description is None:
                txt = 'Invalid FASTA format in {} at line {}: sequence line appeared before header.'
                raise ValueError(txt.format(source_label, line_no))
            # Match Biopython's FASTA behavior: whitespace is formatting, not
            # part of the biological sequence (including spaces and tabs).
            sequence_parts.append(''.join(line.split()))
    if description is not None:
        records.append(FastaRecord(description, ''.join(sequence_parts)))
    return records


def records_to_dict(records: Iterable[FastaRecord], key: str = 'description') -> dict[str, str]:
    """Convert records to a dict and reject duplicate selected keys."""

    if key not in {'description', 'id'}:
        raise ValueError("FASTA record key should be 'description' or 'id'.")
    seq_dict: dict[str, str] = {}
    for record in records:
        record_key = record.description if key == 'description' else record.id
        if record_key in seq_dict:
            raise ValueError('Duplicate FASTA header "{}" found.'.format(record_key))
        seq_dict[record_key] = record.sequence
    return seq_dict


def write_fasta_records(
    records: Iterable[FastaRecord], destination: Any, line_width: int = 60
) -> int:
    """Write FASTA records using deterministic Unix newlines and wrapping."""

    line_width = int(line_width)
    if line_width <= 0:
        raise ValueError('FASTA line_width should be > 0.')
    count = 0
    with _open_text_destination(destination) as handle:
        for record in records:
            description = str(record.description).strip()
            if (description == '') or ('\n' in description) or ('\r' in description):
                raise ValueError('FASTA descriptions should be non-empty single lines.')
            sequence = str(record.sequence)
            handle.write('>{}\n'.format(description))
            if sequence == '':
                handle.write('\n')
            else:
                for start in range(0, len(sequence), line_width):
                    handle.write(sequence[start:start + line_width] + '\n')
            count += 1
    return count


def write_fasta_dict(
    sequences: Mapping[str, str], destination: Any, line_width: int = 60
) -> int:
    records = [FastaRecord(str(name), str(seq)) for name, seq in sequences.items()]
    return write_fasta_records(records, destination, line_width=line_width)
