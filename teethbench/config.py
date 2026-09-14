"""Lightweight model selection, deterministic cohorts and immutable run plans."""
import hashlib
import json
from pathlib import Path
import subprocess

SCHEMA = 'teethbench-v2'
FAMILIES = {
    'sam1': {'base':'sam1_vit_b','large':'sam1_vit_l','huge':'sam1_vit_h'},
    'sam2': {s:'sam2_'+s for s in ['tiny','small','base_plus','large']},
    'sam2.1': {s:'sam21_'+s for s in ['tiny','small','base_plus','large']},
    'sam3': {'default':'sam3'},
}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def select_models(registry, models=None, family=None, size=None):
    if models:
        if family or size:
            raise ValueError('Use either --models or --family with --size, not both.')
        if models==['all']:
            selected=list(registry)
        elif models==['ungated']:
            selected=[n for n,s in registry.items() if not s.get('hf_repo')]
        else:
            selected=models
    else:
        if family=='sam3' and size is None:
            size='default'
        if not family or not size:
            raise ValueError('Specify --models ID [ID ...], or both --family and --size. See the models command.')
        if size not in FAMILIES[family]:
            raise ValueError(f'{family} has no {size} checkpoint. Available sizes: '+', '.join(FAMILIES[family]))
        selected=[FAMILIES[family][size]]
    unknown=[name for name in selected if name not in registry]
    if unknown:
        raise ValueError('Unknown model(s): '+', '.join(unknown)+'. Run: benchmark.py models')
    if len(selected)!=len(set(selected)):
        raise ValueError('Duplicate model selections are not allowed.')
    return selected


def sample_images(images, split, samples, seed):
    if samples<0:
        raise ValueError('--samples must be zero (all) or a positive image count.')
    images=[im for im in images if im['evaluable'] and (split=='all' or im['split']==split)]
    if samples>len(images):
        raise ValueError(f'--samples {samples} exceeds the {len(images)} evaluable images in split {split}.')
    groups={}
    for im in images:
        groups.setdefault(im['pixel_sha256'],[]).append(im)
    ordered=sorted(groups,key=lambda key:hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest())
    limit=samples or len(images)
    selected=[]
    for key in ordered:
        group=sorted(groups[key],key=lambda im:im['id'])
        if len(selected)+len(group)<=limit:
            selected.extend(group)
        if len(selected)==limit:
            break
    if len(selected)!=limit:
        raise ValueError('This sample count would split a duplicate-image group. Choose another count.')
    if not selected:
        raise ValueError('No evaluable images selected.')
    return selected


def source_hashes(root):
    paths=[root/'benchmark.py',root/'common.py',root/'run.py',root/'models.json']
    paths+=sorted((root/'teethbench').glob('*.py'))
    return {str(p.relative_to(root)):digest(p) for p in paths}


def repository_state(root):
    result={}
    for name in ['segment-anything','sam2']:
        command=['git','-C',str(root/'repos'/name)]
        commit=subprocess.check_output(command+['rev-parse','HEAD'],text=True).strip()
        diff=subprocess.check_output(command+['diff','HEAD','--binary'])
        result[name]={'commit':commit,'tracked_diff_sha256':hashlib.sha256(diff).hexdigest()}
    return result


def build_plan(args):
    root=args.root.resolve()
    registry=read(root/'models.json')
    models=select_models(registry,args.models,args.family,args.size)
    if args.mode=='auto' and args.box_condition!='exact':
        raise ValueError('--box-condition applies only to bbox or both mode.')
    images=sample_images(read(root/'prepared/manifest.json')['images'],args.split,args.samples,args.seed)
    checkpoints={}
    checkpoint_files={}
    for name in models:
        path=root/'checkpoints'/registry[name]['checkpoint']
        record=root/'checkpoints'/f'{name}.json'
        if not path.is_file() or not record.is_file():
            raise ValueError(f'Missing checkpoint for {name}. Run: .venv/bin/python download_checkpoints.py --models {name}')
        metadata=read(record)
        checkpoints[name]=metadata['sha256']
        if registry[name].get('hf_repo'):
            if metadata.get('hf_revision')!=registry[name]['hf_revision']:
                raise ValueError(f'Checkpoint revision changed for {name}; run download_checkpoints.py --models {name}.')
            expected={str(Path(registry[name]['checkpoint']).parent / f) for f in registry[name]['hf_files']}
            files=metadata.get('files_sha256',{})
            if set(files)!=expected:
                raise ValueError(f'Incomplete checkpoint metadata for {name}; run download_checkpoints.py --models {name}.')
            if files[registry[name]['checkpoint']]!=metadata['sha256']:
                raise ValueError(f'Inconsistent checkpoint hashes for {name}; run download_checkpoints.py --models {name}.')
            checkpoint_files.update(files)
    return {'schema':SCHEMA,'root':str(root),'models':models,
            'modes':['auto','bbox'] if args.mode=='both' else [args.mode],
            'execution':{'schedule':'model_then_image','mode_order':['bbox','auto'] if args.mode=='both' else [args.mode],
                         'embedding_cache':'one image/crop; shared between modes',
                         'timing':'per-mode encoding inclusive; actual inference recorded separately'},
            'split':args.split,'requested_samples':args.samples,'seed':args.seed,
            'images':images,'image_count':len(images),
            'image_group_count':len({im['pixel_sha256'] for im in images}),
            'tooth_count':sum(im['tooth_count'] for im in images),
            'device':args.device,'threads':args.threads,'precision':'float32',
            'box_condition':args.box_condition,'matching_iou':args.match_iou,
            'bootstrap':args.bootstrap,'boundary_tolerance_px':2.0,
            'auto':{'points_per_side':args.points_per_side,'points_per_batch':args.points_per_batch,
                    'pred_iou_thresh':args.pred_iou_thresh,'stability_score_thresh':args.stability_thresh,
                    'stability_score_offset':1.0,'box_nms_thresh':args.nms_thresh,
                    'crop_n_layers':args.crop_layers,'crop_nms_thresh':args.nms_thresh,
                    'crop_overlap_ratio':512/1500,'crop_n_points_downscale_factor':1,
                    'min_mask_region_area':0,'output_mode':'coco_rle'},
            'manifest_sha256':digest(root/'prepared/manifest.json'),
            'checkpoints_sha256':checkpoints,'checkpoint_files_sha256':checkpoint_files,
            'model_specs':{name:registry[name] for name in models},'source_sha256':source_hashes(root),
            'repositories':repository_state(root)}


def verify_plan(plan):
    root=Path(plan['root'])
    if plan.get('schema')!=SCHEMA:
        raise ValueError('This is not a v2 CLI run directory. Legacy runs use suite.py/status.py.')
    if source_hashes(root)!=plan['source_sha256']:
        raise ValueError('Benchmark source changed since this run started. Restore its source_snapshot to resume, or choose a new output directory.')
    if digest(root/'prepared/manifest.json')!=plan['manifest_sha256']:
        raise ValueError('Dataset manifest changed. Use a new output directory.')
    if repository_state(root)!=plan['repositories']:
        raise ValueError('Official SAM repository code changed. Restore the recorded revisions or use a new output directory.')
