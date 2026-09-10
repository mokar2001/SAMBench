"""Sequential, resumable suite with process isolation and explicit status."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from common import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--models', nargs='+', default=['all'])
    p.add_argument('--split', choices=['dev','test','all'], default='test')
    p.add_argument('--condition', choices=['exact','pad5','pad10','jitter5','jitter10'], default='exact')
    p.add_argument('--seed', type=int, default=20260909)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    # Small models first give useful coverage sooner on CPU. Architecture names stay explicit.
    names = ['sam2_tiny','sam21_tiny','sam1_vit_b','sam2_small','sam21_small',
             'sam2_base_plus','sam21_base_plus','sam2_large','sam21_large','sam1_vit_l','sam1_vit_h'] if args.models==['all'] else args.models
    registry = json.loads((root/'models.json').read_text())
    if any(n not in registry for n in names) or len(set(names)) != len(names):
        raise ValueError('Invalid or duplicate model selection')
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'suite.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = {'models':names,'split':args.split,'condition':args.condition,'seed':args.seed,
            'limit':args.limit,'device':args.device,'threads':args.threads}
    if (out/'suite.json').exists() and json.loads((out/'suite.json').read_text())!=spec:
        raise RuntimeError('Existing suite specification differs')
    save_json(out/'suite.json',spec)
    failures = []
    for name in names:
        command = [sys.executable,str(root/'run.py'),'--root',str(root),'--model',name,'--split',args.split,
                   '--condition',args.condition,'--seed',str(args.seed),'--limit',str(args.limit),
                   '--device',args.device,'--threads',str(args.threads),'--output',str(out/name)]
        print('Starting '+name,flush=True)
        with (out/f'{name}.log').open('a') as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            failures.append(name)
        print(f'{name} exit={result.returncode}',flush=True)
        subprocess.run([sys.executable,str(root/'aggregate.py'),'--suite',str(out)],check=True)
        save_json(out/'suite_status.json',{'finished':False,'last_model':name,'failed_models':failures})
    save_json(out/'suite_status.json',{'finished':True,'complete':not failures,'failed_models':failures})
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
