#!/usr/bin/env python3
"""Check documented CLI syntax and local links without executing examples."""

import argparse
from collections import Counter
import contextlib
import io
from pathlib import Path
import re
import shlex
import sys
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[2]
FENCES = re.compile(r'^```[^\n]*\n(.*?)^```', re.MULTILINE | re.DOTALL)
LINKS = re.compile(r'!?\[[^\]]*\]\(([^\s)]+)|(?:src|href)=[\"\']([^\"\']+)')


def example_commands(body):
    for block in FENCES.finditer(body):
        first_line = body[:block.start(1)].count('\n') + 1
        code = block.group(1)
        # Expand only literal shell-loop values; never read the environment or
        # execute shell substitutions. One representative value checks syntax.
        variables = {}
        for match in re.finditer(r'^\s*for (\w+) in ([\w .-]+); do\s*$', code, re.MULTILINE):
            values = shlex.split(match[2])
            if values:
                variables[match[1]] = values[0]
        code = re.sub(r'\\\s*\n', ' ', code)
        for line in code.splitlines():
            line = line.strip().removeprefix('$ ')
            if not re.match(r'^(?:csubst|python(?:3)? -m csubst)(?:\s|$)', line):
                continue
            for name, value in variables.items():
                line = re.sub(r'\$\{' + name + r'\}|\$' + name + r'\b', value, line)
            yield first_line, line


def check_commands(body, parser, label, counts, errors):
    for line_number, command in example_commands(body):
        location = '{}:{}'.format(label, line_number)
        try:
            argv = shlex.split(command, comments=True)
        except ValueError as exc:
            errors.append('{}: {}'.format(location, exc))
            continue
        argv = argv[1:] if argv[0] == 'csubst' else argv[3:]
        if (argv and argv[0] == 'SUBCOMMAND') or any('$' in token for token in argv):
            counts['templates'] += 1
            continue
        argv = ['--help' if token == '--help-advanced' else token for token in argv]
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                parser.parse_args(argv)
        except SystemExit as exc:
            if exc.code:
                errors.append('{}: {}'.format(location, output.getvalue().strip().splitlines()[-1]))
                continue
        counts['help' if any(a in argv for a in ('-h', '--help', '--version')) else 'commands'] += 1


def check_links(body, path, wiki_dir, label, counts, errors):
    for match in LINKS.finditer(FENCES.sub('', body)):
        link = (match[1] or match[2]).strip('<>')
        url = urlsplit(link)
        if url.netloc == 'github.com' and url.path.startswith('/kfuku52/csubst/wiki/'):
            if wiki_dir is None:
                counts['unchecked_wiki_links'] += 1
                continue
            page = unquote(url.path.split('/wiki/', 1)[1]).rstrip('/')
            target = wiki_dir / (page + '.md')
        elif url.scheme or url.netloc or not url.path:
            continue  # Remote URLs and heading fragments are outside this check.
        else:
            target = path.parent / unquote(url.path)
        counts['links'] += 1
        if not target.exists():
            errors.append('{}: missing link target {}'.format(label, link))


def main():
    args_parser = argparse.ArgumentParser(description=__doc__)
    args_parser.add_argument('--wiki-dir', type=Path, help='Optional local clone of csubst.wiki.git')
    args = args_parser.parse_args()
    if args.wiki_dir is not None and not (args.wiki_dir / 'Home.md').is_file():
        args_parser.error('--wiki-dir must contain the checked-out Wiki, including Home.md')

    # The CLI parser imports only stdlib modules, so CI needs no scientific
    # dependencies or compiled extensions for this check.
    sys.path.insert(0, str(REPO_ROOT))
    from csubst.cli import _build_parser

    parser = _build_parser(show_advanced=True)
    paths = [REPO_ROOT / name for name in ('README.md', 'CONTRIBUTING.md', 'TESTING.md', 'RELEASING.md')]
    paths += sorted((REPO_ROOT / 'docs').rglob('*.md'))
    if args.wiki_dir is not None:
        paths += sorted(args.wiki_dir.glob('*.md'))
    counts = Counter()
    errors = []
    for path in paths:
        body = path.read_text(encoding='utf-8')
        label = ('wiki/' + path.name) if path.parent == args.wiki_dir else str(path.relative_to(REPO_ROOT))
        check_commands(body, parser, label, counts, errors)
        check_links(body, path, args.wiki_dir, label, counts, errors)
    if errors:
        print('Documentation checks failed:\n- ' + '\n- '.join(errors), file=sys.stderr)
        return 1
    print('Documentation checks passed: {} documents, {} commands, {} help examples, '
          '{} templates, {} local links.'.format(len(paths), counts['commands'], counts['help'],
                                               counts['templates'], counts['links']))
    if counts['unchecked_wiki_links']:
        print('Pass --wiki-dir to also check Wiki pages and links ({} references not checked).'
              .format(counts['unchecked_wiki_links']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
