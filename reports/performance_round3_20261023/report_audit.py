"""Check saved evidence, local links, and the product sources actually measured."""

import ast
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def reject_constant(value):
    raise ValueError('Nonstandard JSON constant: ' + value)


def main():
    counts = {'json': 0, 'python': 0, 'local_links': 0, 'product_hashes': 0}
    private_path = re.compile(r'/(?:Users|home)/[^/\s]+')
    for path in HERE.iterdir():
        if not path.is_file():
            continue
        assert path.stat().st_size <= 5 * 1024**2, path.name
        content = path.read_text()
        assert not private_path.search(content), path.name
        if path.suffix == '.json':
            json.loads(content, parse_constant=reject_constant)
            counts['json'] += 1
        elif path.suffix == '.py':
            ast.parse(content, filename=path.name)
            counts['python'] += 1
        elif path.suffix == '.md':
            for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', content):
                if '://' in target or target.startswith('#'):
                    continue
                target = target.split('#', 1)[0]
                assert (path.parent / target).exists(), (path.name, target)
                counts['local_links'] += 1
    data = json.loads((HERE / 'final_cli.json').read_text())
    assert len(data['cases']) == 6
    for name, expected in data['candidate_source'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
        counts['product_hashes'] += 1
    memory = json.loads((HERE / 'memory_followup.json').read_text())
    for case, source in (('conservation', 'csubst/substitution_scan.py'),
                         ('combinations', 'csubst/combination.py')):
        assert all(sample['source_sha256'] == data['candidate_source'][source]
                   for sample in memory['cases'][case]['runs']['after'])
    (HERE / 'report_audit.json').write_text(json.dumps(counts, indent=2) + '\n')
    print('Verified report evidence:', counts)


if __name__ == '__main__':
    main()
