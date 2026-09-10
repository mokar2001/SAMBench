"""Resumable box-prompt inference. A prediction is tied to exactly one GT object."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
import traceback
import numpy as np
from PIL import Image
from common import decode, encode, metrics, prompt_box, save_json, sha256


def create_predictor(spec, checkpoint, device):
    if spec['family'] == 'sam1':
        from segment_anything import SamPredictor, sam_model_registry
        model = sam_model_registry[spec['architecture']](checkpoint=str(checkpoint))
        model.to(device).eval()
        return SamPredictor(model)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(spec['config'], str(checkpoint), device=device,
                       apply_postprocessing=False)
    return SAM2ImagePredictor(model, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--model', required=True)
    p.add_argument('--split', choices=['dev', 'test', 'all'], default='test')
    p.add_argument('--condition', choices=['exact', 'pad5', 'pad10', 'jitter5', 'jitter10'], default='exact')
    p.add_argument('--seed', type=int, default=20260909)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but no CUDA GPU is available')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root = args.root.resolve()
    registry = json.loads((root/'models.json').read_text())
    spec = registry[args.model]
    checkpoint = root/'checkpoints'/spec['checkpoint']
    expected_sha = json.loads((root/'checkpoints'/f'{args.model}.json').read_text())['sha256']
    actual_sha = sha256(checkpoint)
    if actual_sha != expected_sha:
        raise RuntimeError('Checkpoint SHA256 mismatch')
    manifest = json.loads((root/'prepared/manifest.json').read_text())
    images = [im for im in manifest['images'] if im['evaluable'] and (args.split == 'all' or im['split'] == args.split)]
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise ValueError('No selected images')
    out = args.output.resolve()
    out.mkdir(exist_ok=True, parents=True)
    lock = (out/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    revision = {}
    for repo in ['segment-anything','sam2']:
        revision[repo] = subprocess.check_output(['git','-C',str(root/'repos'/repo),'rev-parse','HEAD'], text=True).strip()
    cpu_name = platform.processor()
    if not cpu_name and Path('/proc/cpuinfo').exists():
        cpu_name = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                         if line.startswith('model name')), platform.machine())
    metadata = {'model': args.model, 'model_spec': spec, 'checkpoint_sha256': actual_sha,
                'manifest_sha256': sha256(root/'prepared/manifest.json'),
                'splits_sha256': sha256(root/'prepared/splits.json'),
                'source_sha256': {n: sha256(root/n) for n in ['common.py','run.py']},
                'image_ids': [im['id'] for im in images],
                'image_groups': [im['pixel_sha256'] for im in images],
                'expected_instances': sum(im['tooth_count'] for im in images),
                'condition': args.condition, 'seed': args.seed, 'split': args.split,
                'device': args.device, 'device_name': torch.cuda.get_device_name() if args.device == 'cuda' else cpu_name,
                'threads': args.threads, 'precision': 'float32', 'torch': torch.__version__,
                'numpy': np.__version__, 'python': platform.python_version(), 'host': platform.node(),
                'revisions': revision, 'multimask_output': False, 'postprocessing': False,
                'threshold_logits': 0.0, 'boundary_tolerance_px': 2.0,
                'task': 'one GT-derived box -> one instance mask; original-resolution scoring',
                'warmup': 'first selected image and first box, repeated before timed inference'}
    meta_path = out/'run.json'
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise RuntimeError('Run configuration changed; use a new output directory')
    save_json(meta_path, metadata)
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    def sync():
        if args.device == 'cuda':
            torch.cuda.synchronize()
    with torch.inference_mode():
        start = time.perf_counter()
        predictor = create_predictor(spec, checkpoint, args.device)
        parameter_count = sum(p.numel() for p in predictor.model.parameters())
        sync()
        load_seconds = time.perf_counter()-start
        first = images[0]
        warm = np.asarray(Image.open(root/first['file_name']).convert('RGB'))
        ann = json.loads((root/f'prepared/instances/{first["id"]:04d}.json').read_text())['annotations'][0]
        predictor.set_image(warm)
        predictor.predict(box=prompt_box(ann['box_xyxy'], first['width'], first['height'],
                                         args.condition, ann['id'], args.seed), multimask_output=False)
        sync()
        print(f'Model ready: {args.model}; {len(images)} images; {metadata["expected_instances"]} boxes', flush=True)
        failures = []
        for index, im in enumerate(images, 1):
            target = out/f'images/{im["id"]:04d}.json'
            if target.exists():
                existing = json.loads(target.read_text())
                if existing['image_id'] != im['id'] or len(existing['instances']) != im['tooth_count']:
                    raise RuntimeError('Invalid saved image result')
                continue
            try:
                instances_path = root/f'prepared/instances/{im["id"]:04d}.json'
                if sha256(instances_path) != im['instances_sha256']:
                    raise RuntimeError('Prepared annotation SHA256 mismatch')
                data = json.loads(instances_path.read_text())
                if sha256(root/im['file_name']) != im['image_sha256']:
                    raise RuntimeError('Image SHA256 mismatch')
                rgb = np.asarray(Image.open(root/im['file_name']).convert('RGB'))
                sync()
                start = time.perf_counter()
                predictor.set_image(rgb)
                sync()
                encode_seconds = time.perf_counter()-start
                instances = []
                for ann in data['annotations']:
                    box = prompt_box(ann['box_xyxy'], im['width'], im['height'], args.condition, ann['id'], args.seed)
                    sync()
                    start = time.perf_counter()
                    masks, quality, _ = predictor.predict(box=box, multimask_output=False)
                    sync()
                    seconds = time.perf_counter()-start
                    masks = np.asarray(masks)
                    if masks.shape != (1, im['height'], im['width']):
                        raise RuntimeError(f'Expected exactly one mask; got {masks.shape}')
                    prediction = masks[0].astype(bool)
                    record = dict(instance_id=ann['id'], category_id=ann['category_id'],
                                  prompt_box_xyxy=box.tolist(), predicted_quality=float(np.asarray(quality).reshape(-1)[0]),
                                  decode_seconds=seconds, segmentation=encode(prediction))
                    record.update(metrics(prediction, decode(ann['segmentation'])))
                    instances.append(record)
                result = {'image_id': im['id'], 'encode_seconds': encode_seconds, 'instances': instances,
                          'inference_seconds': encode_seconds+sum(row['decode_seconds'] for row in instances)}
                save_json(target, result)
                print(f'{args.model}: {index}/{len(images)} image={im["id"]} teeth={len(instances)} '
                      f'Dice={np.mean([r["dice"] for r in instances]):.4f} inference={result["inference_seconds"]:.1f}s', flush=True)
            except Exception:
                error = {'image_id': im['id'], 'traceback': traceback.format_exc()}
                failures.append(error)
                save_json(out/f'errors/{im["id"]:04d}.json', error)
                print(error['traceback'], flush=True)
        completed = [im['id'] for im in images if (out/f'images/{im["id"]:04d}.json').exists()]
        save_json(out/'status.json', {'complete': len(completed) == len(images), 'completed_images': len(completed),
                                     'expected_images': len(images), 'failed_this_attempt': failures,
                                     'model_load_seconds': load_seconds,
                                     'parameter_count': parameter_count,
                                     'peak_process_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                                     'peak_cuda_allocated_gib': torch.cuda.max_memory_allocated()/1024**3 if args.device == 'cuda' else None})
        if len(completed) != len(images):
            raise RuntimeError('Incomplete run; failed images remain resumable and are not eligible for ranking')


if __name__ == '__main__':
    main()
