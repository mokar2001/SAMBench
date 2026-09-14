"""Download official checkpoints with SHA256 provenance; SAM 3 uses cached HF login."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
from common import save_json
from teethbench.config import select_models


def fetch_checkpoint(root, name, model):
    out=root/'checkpoints'
    out.mkdir(exist_ok=True,parents=True)
    target=out/model['checkpoint']
    extra={}
    if model.get('hf_repo'):
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import HfHubHTTPError
        except ImportError as exc:
            raise ValueError('Install requirements-sam3.txt before downloading SAM 3.') from exc
        files={}
        for filename in model['hf_files']:
            try:
                path=Path(hf_hub_download(model['hf_repo'],filename,revision=model['hf_revision'],
                                         local_dir=target.parent))
            except HfHubHTTPError as exc:
                raise ValueError('SAM 3 download failed. Run hf auth login on this machine using an account approved for facebook/sam3; check read access and connectivity.') from exc
            files[str(path.resolve().relative_to(out.resolve()))]=sha256(path)
        extra['files_sha256']=files
    elif not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        part=target.with_suffix(target.suffix+'.part')
        subprocess.run(['curl','-fLsS','--retry','4','--connect-timeout','30','-C','-',
                        '-o',str(part),model['url']],check=True)
        part.replace(target)
    checkpoint_hash=extra.get('files_sha256',{}).get(str(target.resolve().relative_to(out.resolve())))
    record=dict(model,name=name,sha256=checkpoint_hash or sha256(target),bytes=target.stat().st_size,**extra)
    save_json(out/(name+'.json'),record)
    print(json.dumps(record),flush=True)
    return record


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--models', nargs='+', default=['ungated'],help='IDs, all (includes gated SAM 3), or ungated (default: SAM 1/2/2.1)')
    args = p.parse_args()
    registry = json.loads((args.root / 'models.json').read_text())
    try:
        names=select_models(registry,models=args.models)
    except ValueError as exc:
        p.error(str(exc))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda name:fetch_checkpoint(args.root.resolve(),name,registry[name]),names))


if __name__ == '__main__':
    main()
