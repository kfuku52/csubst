#!/usr/bin/env python3
"""Install exactly one freshly built artifact with the requested test tools."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdist', action='store_true')
    parser.add_argument('--extra', choices=['test', 'dev'], default='test')
    args = parser.parse_args()
    pattern = 'csubst-*.tar.gz' if args.sdist else 'from-sdist/csubst-*.whl'
    artifacts = list(Path('dist').glob(pattern))
    if len(artifacts) != 1:
        raise RuntimeError('Expected exactly one artifact matching ' + pattern)
    requirement = str(artifacts[0].resolve()) + '[' + args.extra + ']'
    subprocess.run([sys.executable, '-m', 'pip', 'install', requirement], check=True)


if __name__ == '__main__':
    main()
