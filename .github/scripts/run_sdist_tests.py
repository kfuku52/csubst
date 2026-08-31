#!/usr/bin/env python3
"""Run the complete source-only suite from an extracted source distribution."""

from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def main():
    archives = list(Path('dist').glob('csubst-*.tar.gz'))
    if len(archives) != 1:
        raise RuntimeError('Expected exactly one source distribution')
    with tempfile.TemporaryDirectory(prefix='csubst-sdist-test-') as tmp:
        with tarfile.open(archives[0]) as archive:
            archive.extractall(tmp, filter='data')
        sources = [p.parent for p in Path(tmp).glob('*/pytest.ini')]
        if len(sources) != 1:
            raise RuntimeError('Expected one pytest root inside the source distribution')
        for options in (['-n', 'auto', '--dist', 'worksteal', '-m', 'not process'], ['-m', 'process']):
            subprocess.run([sys.executable, '-m', 'pytest', '-q', *options], cwd=sources[0], check=True)


if __name__ == '__main__':
    main()
