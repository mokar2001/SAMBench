"""One isolated model/mode process; native inference followed by evaluation."""
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


def automatic_generator(model, family, settings):
    if family=='sam1':
        from segment_anything import SamAutomaticMaskGenerator
        return SamAutomaticMaskGenerator(model,**settings)
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    return SAM2AutomaticMaskGenerator(model,mask_threshold=0.0,use_m2m=False,multimask_output=True,**settings)


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


def run_job(output, model_name, mode):
    import torch
    output=Path(output)
    plan=read(output/'plan.json')
    verify_plan(plan)
    root=Path(plan['root'])
    spec=read(root/'models.json')[model_name]
    job=output/mode/model_name
    job.mkdir(parents=True,exist_ok=True)
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
    if spec['family']=='sam3':
        try:
            runtime['packages'].update({name:version(name) for name in ['transformers','huggingface_hub','safetensors']})
        except Exception as exc:
            raise ValueError('Install SAM 3 dependencies using requirements-sam3.txt.') from exc
    if (job/'runtime.json').exists() and read(job/'runtime.json')!=runtime:
        raise ValueError('Runtime/hardware changed; use a new output directory.')
    write(job/'runtime.json',runtime)
    images=plan['images']
    pending=[im for im in images if not image_complete(job/f'images/{im["id"]:04d}.json',im)]
    if not pending:
        print(f'{model_name} {mode}: all {len(images)} images already complete; skipped.',flush=True)
        return
    write(job/'status.json',{'state':'loading','completed_images':len(images)-len(pending),'expected_images':len(images)})
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
            predictor=Sam3Predictor(checkpoint,device,spec['transformers_version'])
        else:
            predictor=create_predictor(spec,checkpoint,device)
        # Some upstream imports change TF32 globally; enforce the recorded FP32 protocol.
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        sync()
        load_seconds=time.perf_counter()-start
        generator=(sam3_generator(predictor,plan['auto']) if spec['family']=='sam3' else
                   automatic_generator(predictor.model,spec['family'],plan['auto'])) if mode=='auto' else None
        write(job/'status.json',{'state':'warmup','completed_images':len(images)-len(pending),'expected_images':len(images)})
        print(f'{model_name} {mode}: model loaded in {load_seconds:.1f}s; warming up.',flush=True)
        warm_im=pending[0]
        warm=np.array(Image.open(root/warm_im['file_name']).convert('RGB'),copy=True)
        predictor.set_image(warm)
        if mode=='bbox':
            ann=read(root/f'prepared/instances/{warm_im["id"]:04d}.json')['annotations'][0]
            predictor.predict(box=np.array(ann['box_xyxy'],dtype=np.float32),multimask_output=False)
        else:
            # GT-independent warmup: the full image and one image-center point.
            predictor.predict(point_coords=np.array([[warm_im['width']/2,warm_im['height']/2]]),
                              point_labels=np.array([1]),multimask_output=True)
        sync()
        write(job/'model_info.json',{'parameters':sum(p.numel() for p in predictor.model.parameters()),
                                     'load_seconds':load_seconds,'warmup':'one full image and one prompt; auto uses image center',
                                     'backend':spec.get('backend','official_meta'),
                                     'auto_generator':'sam2_amg_with_sam3_predictor' if spec['family']=='sam3' else 'official_meta'})
        failed=[]
        for number,im in enumerate(images,1):
            destination=job/f'images/{im["id"]:04d}.json'
            if image_complete(destination,im):
                continue
            try:
                completed=sum((job/f'images/{i["id"]:04d}.json').exists() for i in images)
                write(job/'status.json',{'state':'running','current_image_id':im['id'],
                                         'completed_images':completed,'expected_images':len(images)})
                path=root/f'prepared/instances/{im["id"]:04d}.json'
                if digest(path)!=im['instances_sha256'] or digest(root/im['file_name'])!=im['image_sha256']:
                    raise ValueError('Input image or annotation hash mismatch')
                rgb=np.array(Image.open(root/im['file_name']).convert('RGB'),copy=True)
                annotations=read(path)['annotations']
                if mode=='auto':
                    # No annotation/box/point derived from ground truth reaches generate().
                    sync()
                    start=time.perf_counter()
                    raw=generator.generate(rgb)
                    sync()
                    seconds=time.perf_counter()-start
                    predictions=normalize_automatic(raw)
                    evaluated=evaluate_auto(predictions,annotations,plan['matching_iou'])
                    result={'image_id':im['id'],'inference_seconds':seconds,'predictions':predictions,**evaluated}
                else:
                    sync()
                    start=time.perf_counter()
                    predictor.set_image(rgb)
                    sync()
                    encoding=time.perf_counter()-start
                    predictions,scored=[],[]
                    decode_total=0.0
                    for ann in annotations:
                        box=prompt_box(ann['box_xyxy'],im['width'],im['height'],plan['box_condition'],ann['id'],plan['seed'])
                        sync()
                        start=time.perf_counter()
                        masks,quality,_=predictor.predict(box=box,multimask_output=False)
                        sync()
                        elapsed=time.perf_counter()-start
                        decode_total+=elapsed
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
                    result={'image_id':im['id'],'inference_seconds':encoding+decode_total,'encode_seconds':encoding,
                            'predictions':predictions,'gt_scores':scored,'summary':summary}
                write(destination,result)
                error_path=job/f'errors/{im["id"]:04d}.json'
                if error_path.exists():
                    error_path.unlink()
                summary=result['summary']
                label=(f'quality={summary["instance_quality"]:.4f} TP={summary["tp"]} FP={summary["fp"]} FN={summary["fn"]}'
                       if mode=='auto' else f'Dice={summary["gt_macro_dice"]:.4f}')
                print(f'{model_name} {mode} [{number}/{len(images)}] image={im["id"]} {label} {result["inference_seconds"]:.1f}s',flush=True)
            except Exception:
                failed.append(im['id'])
                write(job/f'errors/{im["id"]:04d}.json',{'image_id':im['id'],'traceback':traceback.format_exc()})
                print(traceback.format_exc(),flush=True)
            completed=sum((job/f'images/{i["id"]:04d}.json').exists() for i in images)
            write(job/'status.json',{'state':'running','completed_images':completed,'expected_images':len(images),'failed_images':failed})
        write(job/'status.json',{'state':'complete' if not failed else 'failed','completed_images':len(images)-len(failed),
                                 'expected_images':len(images),'failed_images':failed,
                                 'peak_process_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=='Linux' else 1024**3),
                                 'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated()/1024**3 if device=='cuda' else None})
        if failed:
            raise RuntimeError(f'{len(failed)} failed image(s); rerun the same command or use resume.')
