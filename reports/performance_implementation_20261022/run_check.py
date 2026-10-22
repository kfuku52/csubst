"""Run one documented check and append its exit status and concise log evidence."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--cwd', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    started = time.perf_counter()
    with args.log.open('w') as stream:
        result = subprocess.run(command, cwd=args.cwd, stdout=stream, stderr=subprocess.STDOUT)
    output = args.log.read_text()
    root = str(Path(__file__).resolve().parents[2])
    def clean(text):
        return text.replace(root, '${REPOSITORY}').replace(str(Path.home()), '${HOME}')
    records = json.loads(args.record.read_text()) if args.record.exists() else []
    record = dict(label=args.label, command=[clean(value) for value in command],
                  cwd=clean(str(args.cwd or Path.cwd())),
                  exit_code=result.returncode, seconds=time.perf_counter() - started,
                  summaries=re.findall(r'^.*(?:passed|failed|skipped|deselected|checks passed|Verified|Success|FAILED|ERROR).*$'
                                       , clean(output), flags=re.MULTILINE)[-50:])
    records.append(record)
    args.record.write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)
    if result.returncode:
        print(clean(output[-6000:]), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
