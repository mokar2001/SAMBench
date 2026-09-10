"""Download only official checkpoints; retain SHA256 provenance. No credentials needed."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--models', nargs='+', default=['all'])
    args = p.parse_args()
    registry = json.loads((args.root / 'models.json').read_text())
    names = list(registry) if args.models == ['all'] else args.models
    out = args.root / 'checkpoints'
    out.mkdir(exist_ok=True, parents=True)
    def fetch(name):
        model = registry[name]
        target = out / model['checkpoint']
        if not target.exists():
            part = target.with_suffix(target.suffix + '.part')
            subprocess.run(['curl', '-fLsS', '--retry', '4', '--connect-timeout', '30',
                            '-C', '-', '-o', str(part), model['url']], check=True)
            part.replace(target)
        record = dict(model, name=name, sha256=sha256(target), bytes=target.stat().st_size)
        (out / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, names))


if __name__ == '__main__':
    main()
