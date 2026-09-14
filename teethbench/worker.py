"""One isolated model process; reuse each image across the requested modes."""
from pathlib import Path
import platform
import resource
import time
import traceback
from importlib.metadata import version
import numpy as np
from PIL import Image
from pycocotools import mask as mu
from common import decode,encode,metrics,prompt_box
from run import create_predictor
from .config import digest,read,write,verify_plan
from .evaluation import evaluate_auto
from .embeddings import ImageEmbeddingCache


def automatic_generator(model, family, settings):
    if family=='sam1':
        from segment_anything import SamAutomaticMaskGenerator
        generator=SamAutomaticMaskGenerator(model.model,**settings)
    else:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        generator=SAM2AutomaticMaskGenerator(model.model,mask_threshold=0.0,use_m2m=False,multimask_output=True,**settings)
    generator.predictor=model
    return generator


def normalize_automatic(records):
    result=[]
    for record in records:
        rle=dict(record['segmentation'])
        if isinstance(rle['counts'],bytes):
            rle['counts']=rle['counts'].decode('ascii')
        rle['size']=[int(n) for n in rle['size']]
        quality=float(record['predicted_iou'])
        if not np.isfinite(quality):
            raise ValueError('Non-finite model quality score')
        result.append({'segmentation':rle,'predicted_quality':quality,
                       'stability_score':float(record['stability_score']),
                       'area':int(mu.area(rle)),'bbox_xywh':mu.toBbox(rle).tolist(),
                       'point_coords':record['point_coords'],'crop_box':record['crop_box']})
    return sorted(result,key=lambda p:-p['predicted_quality'])


def image_complete(path, im):
    if not path.exists():
        return False
    saved=read(path)
    if saved['image_id']!=im['id'] or saved['summary']['gt_count']!=im['tooth_count']:
        raise ValueError(f'Invalid saved result: {path}')
    return True


def infer_bbox(predictor, rgb, annotations, im, plan, sync):
    event_start=len(predictor.events)
    sync()
    start=time.perf_counter()
    predictor.set_image(rgb)
    sync()
    actual=time.perf_counter()-start
    predictions,scored=[],[]
    for ann in annotations:
        box=prompt_box(ann['box_xyxy'],im['width'],im['height'],plan['box_condition'],ann['id'],plan['seed'])
        sync()
        start=time.perf_counter()
        masks,quality,_=predictor.predict(box=box,multimask_output=False)
        sync()
        elapsed=time.perf_counter()-start
        actual+=elapsed
        if masks.shape!=(1,im['height'],im['width']):
            raise ValueError(f'Expected one original-resolution mask, got {masks.shape}')
        mask=masks[0].astype(bool)
        scored.append({'instance_id':ann['id'],'category_id':ann['category_id'],
                       'prediction_index':len(predictions),**metrics(mask,decode(ann['segmentation']))})
        predictions.append({'segmentation':encode(mask),'instance_id':ann['id'],
                            'prompt_box_xyxy':box.tolist(),'predicted_quality':float(np.asarray(quality).reshape(-1)[0]),
                            'decode_seconds':elapsed})
    summary={'gt_count':len(annotations),'prediction_count':len(predictions),
             'empty_predictions':sum(r['empty'] for r in scored),
             **{f'gt_macro_{key}':float(np.mean([r[key] for r in scored])) for key in ['dice','iou','boundary_f1']}}
    return {'image_id':im['id'],**predictor.timing(event_start,actual),
            'predictions':predictions,'gt_scores':scored,'summary':summary}


def infer_auto(predictor, generator, rgb, annotations, im, plan, sync):
    # Only prompt-independent image features are shared. No annotation, box, mask,
    # or previous prompt reaches generate(), whose grid and filtering stay native.
    event_start=len(predictor.events)
    sync()
    start=time.perf_counter()
    raw=generator.generate(rgb)
    sync()
    timing=predictor.timing(event_start,time.perf_counter()-start)
    predictions=normalize_automatic(raw)
    evaluated=evaluate_auto(predictions,annotations,plan['matching_iou'])
    return {'image_id':im['id'],**timing,'predictions':predictions,**evaluated}


def run_job(output, model_name, mode):
    import torch
    output=Path(output)
    plan=read(output/'plan.json')
    verify_plan(plan)
    modes=['bbox','auto'] if mode=='both' else [mode]
    if not set(modes)<=set(plan['modes']):
        raise ValueError('Worker modes are not included in the saved plan.')
    root=Path(plan['root'])
    spec=read(root/'models.json')[model_name]
    images=plan['images']
    jobs={m:output/m/model_name for m in modes}
    completed={m:{im['id'] for im in images if image_complete(jobs[m]/f'images/{im["id"]:04d}.json',im)} for m in modes}
    pending=[im for im in images if any(im['id'] not in completed[m] for m in modes)]
    if not pending:
        print(f'{model_name} {mode}: all {len(images)} images already complete; skipped.',flush=True)
        return
    torch.set_num_threads(plan['threads'])
    torch.set_num_interop_threads(1)
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    device=plan['device']
    if device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but no CUDA GPU is available; choose --device cpu.')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    runtime={'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),
             'packages':{name:version(name) for name in ['torchvision','Pillow','scipy','pycocotools']},
             'host':platform.node(),'device':device,'threads':plan['threads'],
             'device_name':torch.cuda.get_device_name() if device=='cuda' else next(
                 (s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                  if s.startswith('model name')),platform.machine()) if Path('/proc/cpuinfo').exists() else platform.machine()}
    if 'sam3' in plan['models']:
        try:
            runtime['packages'].update({name:version(name) for name in ['transformers','huggingface_hub','safetensors']})
        except Exception as exc:
            raise ValueError('Install SAM 3 dependencies using requirements-sam3.txt.') from exc
    active_modes=[m for m in modes if len(completed[m])<len(images)]
    failed={m:[] for m in active_modes}
    def status(m, state, **extra):
        write(jobs[m]/'status.json',{'state':state,'completed_images':len(completed[m]),
              'expected_images':len(images),'failed_images':failed[m],**extra})
    for m in active_modes:
        job=jobs[m]
        if (job/'runtime.json').exists() and read(job/'runtime.json')!=runtime:
            raise ValueError('Runtime/hardware changed; use a new output directory.')
        write(job/'runtime.json',runtime)
        status(m,'loading')
    expected_files=dict(plan.get('checkpoint_files_sha256',{})) if spec['family']=='sam3' else {}
    expected_files[spec['checkpoint']]=plan['checkpoints_sha256'][model_name]
    for filename,expected_hash in expected_files.items():
        if digest(root/'checkpoints'/filename)!=expected_hash:
            raise ValueError(f'Checkpoint/config SHA256 mismatch: {filename}')
    checkpoint=root/'checkpoints'/spec['checkpoint']
    def sync():
        if device=='cuda':
            torch.cuda.synchronize()
    if device=='cuda':
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        start=time.perf_counter()
        if spec['family']=='sam3':
            from .sam3 import Sam3Predictor,automatic_generator as sam3_generator
            native=Sam3Predictor(checkpoint,device,spec['transformers_version'])
        else:
            native=create_predictor(spec,checkpoint,device)
        # Some upstream imports change TF32 globally; enforce the recorded FP32 protocol.
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        sync()
        load_seconds=time.perf_counter()-start
        predictor=ImageEmbeddingCache(native,sync)
        generator=(sam3_generator(predictor,plan['auto']) if spec['family']=='sam3' else
                   automatic_generator(predictor,spec['family'],plan['auto'])) if 'auto' in active_modes else None
        for m in active_modes:
            status(m,'warmup')
        print(f'{model_name} {mode}: model loaded once in {load_seconds:.1f}s; warming up.',flush=True)
        warm_im=pending[0]
        warm=np.array(Image.open(root/warm_im['file_name']).convert('RGB'),copy=True)
        predictor.begin_image()
        predictor.set_image(warm)
        if 'bbox' in active_modes:
            ann=read(root/f'prepared/instances/{warm_im["id"]:04d}.json')['annotations'][0]
            predictor.predict(box=np.array(ann['box_xyxy'],dtype=np.float32),multimask_output=False)
        if 'auto' in active_modes:
            predictor.predict(point_coords=np.array([[warm_im['width']/2,warm_im['height']/2]]),
                              point_labels=np.array([1]),multimask_output=True)
        sync()
        predictor.begin_image()  # Warmup is separate from measured inference.
        del warm
        for m in active_modes:
            write(jobs[m]/'model_info.json',{'parameters':sum(p.numel() for p in native.model.parameters()),
                  'load_seconds':load_seconds,'warmup':'one full image; one box and/or center point for requested modes',
                  'backend':spec.get('backend','official_meta'),
                  'auto_generator':'sam2_amg_with_sam3_predictor' if spec['family']=='sam3' else 'official_meta',
                  'worker_modes':active_modes,'embedding_cache':'one image/crop shared across prompts and modes',
                  'timing':'inference_seconds includes shared encoding per mode; actual_inference_seconds excludes reuse'})
        def record_error(m, im):
            failed[m].append(im['id'])
            write(jobs[m]/f'errors/{im["id"]:04d}.json',{'image_id':im['id'],'traceback':traceback.format_exc()})
            print(traceback.format_exc(),flush=True)
        for number,im in enumerate(images,1):
            image_modes=[m for m in active_modes if im['id'] not in completed[m]]
            if not image_modes:
                continue
            predictor.begin_image()
            try:
                path=root/f'prepared/instances/{im["id"]:04d}.json'
                if digest(path)!=im['instances_sha256'] or digest(root/im['file_name'])!=im['image_sha256']:
                    raise ValueError('Input image or annotation hash mismatch')
                rgb=np.array(Image.open(root/im['file_name']).convert('RGB'),copy=True)
                annotations=read(path)['annotations']
            except Exception:
                for m in image_modes:
                    record_error(m,im)
                    status(m,'running')
                continue
            for m in image_modes:
                status(m,'running',current_image_id=im['id'])
                try:
                    result=(infer_auto(predictor,generator,rgb,annotations,im,plan,sync) if m=='auto' else
                            infer_bbox(predictor,rgb,annotations,im,plan,sync))
                    write(jobs[m]/f'images/{im["id"]:04d}.json',result)
                    completed[m].add(im['id'])
                    error_path=jobs[m]/f'errors/{im["id"]:04d}.json'
                    if error_path.exists():
                        error_path.unlink()
                    summary=result['summary']
                    label=(f'quality={summary["instance_quality"]:.4f} TP={summary["tp"]} FP={summary["fp"]} FN={summary["fn"]}'
                           if m=='auto' else f'Dice={summary["gt_macro_dice"]:.4f}')
                    print(f'{model_name} {m} [{number}/{len(images)}] image={im["id"]} {label} '
                          f'{result["inference_seconds"]:.1f}s (actual {result["actual_inference_seconds"]:.1f}s; '
                          f'encodes={result["encoder_calls"]}, reuses={result["embedding_reuses"]})',flush=True)
                except Exception:
                    record_error(m,im)
                    predictor.reset_predictor()  # Failed modes cannot leave stale features for the next mode.
                status(m,'running')
            predictor.reset_predictor()
        for m in active_modes:
            status(m,'complete' if not failed[m] else 'failed',
                   peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=='Linux' else 1024**3),
                   peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if device=='cuda' else None)
        if any(failed.values()):
            raise RuntimeError(f'{sum(map(len,failed.values()))} failed image/mode(s); use resume to retry missing results.')
