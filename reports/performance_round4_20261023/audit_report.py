"""Validate durable evidence, output audits and the final product fingerprints."""

import hashlib
import json
from pathlib import Path
import re
import subprocess

from variants import diagnostic_source, endpoint_source, scalar_source, selection_source

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def invalid_constant(value):
    raise ValueError('Nonstandard JSON constant: ' + value)


def main():
    files = [p for p in HERE.iterdir() if p.is_file()]
    for path in files:
        assert path.stat().st_size < 5 * 1024**2, path.name
        assert str(Path.home()) not in path.read_text(), path.name
        if path.suffix == '.json':
            json.loads(path.read_text(), parse_constant=invalid_constant)
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', (HERE / 'README.md').read_text()):
        assert (HERE / target).is_file(), target
    product = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((ROOT / 'csubst').glob('*.py'))}
    changed = {'csubst/endpoint.py', 'csubst/substitution_scan.py', 'csubst/scan_statistics.py'}
    pairs = 0
    for name, audit, expected_cases in (
        ('cli.json', 'output_audit.json', 5),
        ('cli_followup.json', 'output_audit_followup.json', 3),
    ):
        data = json.loads((HERE / name).read_text())
        assert data['candidate_source'] == product, name
        assert {key for key in product if data['baseline_source'][key] != product[key]} == changed, name
        assert len(data['cases']) == expected_cases, name
        expected_pairs = 0
        for case in data['cases'].values():
            assert len(case['runs']['before']) == len(case['runs']['after']) == 5
            assert len(case['comparisons']) == 6
            expected_pairs += len(case['comparisons'])
        audited = json.loads((HERE / audit).read_text())
        assert audited['paired_runs_including_warmups'] == expected_pairs, name
        assert audited['cases'].keys() == data['cases'].keys(), name
        pairs += expected_pairs
    versions = json.loads((HERE / 'environment.json').read_text())
    assert versions['product_sha256'] == {key: product[key] for key in changed}
    for name, transform in (
        ('endpoint.py', endpoint_source),
        ('substitution_scan.py', lambda source: selection_source(diagnostic_source(source))),
        ('scan_statistics.py', scalar_source),
    ):
        baseline = subprocess.check_output(
            ['git', 'show', versions['baseline_commit'] + ':csubst/' + name], cwd=ROOT, text=True)
        assert transform(baseline) == (ROOT / 'csubst' / name).read_text(), name
    subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
    print(f'Verified {len(files)} report files, {pairs} CLI output pairs and final source fingerprints.')


if __name__ == '__main__':
    main()
