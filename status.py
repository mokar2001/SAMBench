"""Read progress without loading PyTorch or doing inference."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('suite',type=Path)
args=p.parse_args()
spec=json.loads((args.suite/'suite.json').read_text())
for name in spec['models']:
    folder=args.suite/name
    if not (folder/'run.json').exists():
        print(f'{name:20} queued')
        continue
    meta=json.loads((folder/'run.json').read_text())
    done=sum((folder/f'images/{i:04d}.json').exists() for i in meta['image_ids'])
    status='complete' if done==len(meta['image_ids']) else 'incomplete/running'
    print(f'{name:20} {done:4}/{len(meta["image_ids"])} images {status}')
