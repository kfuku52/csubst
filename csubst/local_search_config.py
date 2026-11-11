"""Local structure search database prefixes and the files they refer to."""

from glob import escape, iglob
import os
from pathlib import Path
import re
import shlex
from collections.abc import Iterator
from typing import Any


UNIPROT_DATABASES = frozenset({'swissmodel', 'alphafill', 'alphafold'})
DATABASE_PATH_OPTIONS = frozenset({'blastp_db', 'mmseqs_db'})
_BLAST_SUFFIX = re.compile(r'(?:\.\d+)?\.p[a-z]{2}')
_MMSEQS_SUFFIX = re.compile(r'(?:_h)?(?:\.idx)?(?:\.\d+)?(?:\.(?:index|dbtype|lookup|source))?')


def normalize_paths(config: dict[str, Any], databases: str) -> None:
    names = {name.strip().lower() for name in databases.split(',')}
    if not (names & (UNIPROT_DATABASES | {'pdb'})):
        raise ValueError('--database_search local requires at least one supported structure database.')
    for option, required in (('blastp_db', bool(names & UNIPROT_DATABASES)), ('mmseqs_db', 'pdb' in names)):
        value = str(config.get(option) or '').strip()
        if required and not value:
            raise ValueError('--database_search local requires --{} for the selected databases.'.format(option))
        if value:
            config[option] = os.path.abspath(os.path.expanduser(value))


def _blast_alias_databases(alias: Path) -> Iterator[Path]:
    try:
        with alias.open(encoding='utf-8', errors='replace') as handle:
            for line in handle:
                fields = line.strip().split(maxsplit=1)
                if len(fields) == 2 and fields[0] == 'DBLIST':
                    for prefix in shlex.split(fields[1], comments=True):
                        yield alias.parent / prefix
    except (OSError, ValueError):
        # BLAST reports unreadable or malformed aliases when the search runs.
        return


def database_files(option: str, value: str) -> Iterator[str]:
    """Enumerate existing sidecars before the CLI opens any log or output."""
    prefixes = [Path(os.path.abspath(os.path.expanduser(value)))]
    seen = set()
    seen_aliases = set()
    while prefixes:
        prefix = Path(os.path.abspath(prefixes.pop()))
        if prefix in seen:
            continue
        seen.add(prefix)
        prefix_text = str(prefix)
        if option == 'blastp_db':
            for name in iglob(escape(prefix_text) + '.*'):
                if _BLAST_SUFFIX.fullmatch(name[len(prefix_text):]):
                    yield name
            alias = prefix.with_name(prefix.name + '.pal')
            if alias.is_file():
                # Resolve symlinks and dot segments for cycle detection while
                # keeping the supplied directory for relative DBLIST paths.
                alias_target = os.path.realpath(alias)
                if alias_target not in seen_aliases:
                    seen_aliases.add(alias_target)
                    prefixes.extend(_blast_alias_databases(alias))
        else:
            for name in iglob(escape(prefix_text) + '*'):
                if _MMSEQS_SUFFIX.fullmatch(name[len(prefix_text):]):
                    yield name
