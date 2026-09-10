"""Convert original Supervisely polygons to independent COCO RLE tooth instances."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from common import box_from_mask, encode, polygon_mask, save_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--reuse-validated-polygons', action='store_true', help='Reuse this pipeline\'s existing masks after matching original file hashes')
    args = parser.parse_args()
    root = args.root.resolve()
    src = root / 'raw/Teeth Segmentation JSON'
    out = root / 'prepared'
    out.mkdir(exist_ok=True)
    annotation_paths = sorted(src.glob('*/ann/*.json'))
    if len(annotation_paths) != 598:
        raise ValueError(f'Expected 598 annotations; found {len(annotation_paths)}')
    images, annotations, rows, warnings, failures = [], [], [], [], []
    class_counts, groups = Counter(), defaultdict(list)
    png_iou = []
    invalid_objects, empty_images = [], []
    raw_objects = 0
    known_zero_area = {1166088059,1166088743,1166097782,1166099622,1166090241}
    for image_id, ann_path in enumerate(annotation_paths, 1):
        ann = json.loads(ann_path.read_text())
        raw_objects += len(ann['objects'])
        image_path = ann_path.parent.parent / 'img' / ann_path.name.removesuffix('.json')
        try:
            cached = None
            cached_path = out / f'instances/{image_id:04d}.json'
            if args.reuse_validated_polygons and cached_path.exists():
                candidate = json.loads(cached_path.read_text())
                if (candidate['image']['source_annotation_sha256'] == sha256(ann_path)
                    and candidate['image']['image_sha256'] == sha256(image_path)):
                    cached = candidate
            cached_objects = {o['source_object_id']: o for o in cached['annotations']} if cached else {}
            im = Image.open(image_path).convert('RGB')
            width, height = im.size
            if ann['size'] != {'height': height, 'width': width}:
                raise ValueError('Annotation and image sizes differ')
            pixels = np.asarray(im)
            pixel_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
            groups[pixel_hash].append(image_id)
            instances, seen = [], Counter()
            occupancy = np.zeros((height, width), dtype=np.uint8)
            for obj in ann['objects']:
                if obj['geometryType'] != 'polygon':
                    raise ValueError(f'Unsupported geometry: {obj["geometryType"]}')
                category = int(obj['classTitle'])
                if not 1 <= category <= 32:
                    raise ValueError(f'Unexpected class {category}')
                if obj['id'] in cached_objects:
                    previous = cached_objects[obj['id']]
                    box, rle, area = previous['box_xyxy'], previous['segmentation'], previous['area']
                else:
                    try:
                        mask = polygon_mask(obj['points'], height, width)
                    except ValueError as exc:
                        if str(exc) == 'Empty polygon mask' and obj['id'] in known_zero_area:
                            invalid_objects.append({'image_id':image_id,'source_object_id':obj['id'],
                                'source_annotation':str(ann_path.relative_to(root)), 'classTitle':obj['classTitle'],
                                'reason':'verified_zero_area_polygon','points':obj['points']})
                            continue
                        raise
                    box, rle, area = box_from_mask(mask), encode(mask), int(mask.sum())
                    occupancy += mask.astype(np.uint8)
                coords = np.asarray(obj['points']['exterior'])
                if ((coords < 0).any() or (coords[:, 0] > width).any() or (coords[:, 1] > height).any()):
                    warnings.append({'image_id': image_id, 'object_id': obj['id'], 'reason': 'polygon_clipped_to_image'})
                instance = {'id': len(annotations)+1, 'image_id': image_id,
                            'category_id': category, 'source_object_id': obj['id'],
                            'segmentation': rle, 'bbox': [box[0], box[1], box[2]-box[0], box[3]-box[1]],
                            'box_xyxy': box, 'area': area, 'iscrowd': 0}
                annotations.append(instance)
                instances.append(instance)
                seen[category] += 1
                class_counts[category] += 1
                rows.append({'image_id': image_id, 'instance_id': instance['id'],
                             'source_object_id': obj['id'], 'image_path': str(image_path.relative_to(root)),
                             'tooth_class': category, 'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3],
                             'width': width, 'height': height, 'area_pixels': instance['area']})
            if not instances:
                empty_images.append({'image_id':image_id,'file_name':str(image_path.relative_to(root)),
                                     'reason':'no_ground_truth_teeth_no_box_prompt_possible'})
            duplicates = {str(k): v for k, v in seen.items() if v > 1}
            if duplicates:
                warnings.append({'image_id': image_id, 'reason': 'repeated_tooth_class', 'counts': duplicates})
            image = {'id': image_id, 'file_name': str(image_path.relative_to(root)),
                     'source_annotation': str(ann_path.relative_to(root)),
                     'source_annotation_sha256': sha256(ann_path), 'image_sha256': sha256(image_path),
                     'pixel_sha256': pixel_hash, 'width': width, 'height': height,
                     'tooth_count': len(instances), 'evaluable':bool(instances),
                     'overlap_pixels': cached['image']['overlap_pixels'] if cached else int((occupancy > 1).sum())}
            machine = root / 'raw/Teeth Segmentation PNG' / ann_path.parent.parent.name / 'masks_machine' / (image_path.stem+'.png')
            if cached and 'polygon_vs_png_union_iou' in cached['image']:
                score = cached['image']['polygon_vs_png_union_iou']
                png_iou.append(score)
                image['polygon_vs_png_union_iou'] = score
            elif machine.exists():
                exported = np.array(Image.open(machine))
                if exported.ndim == 3:
                    exported = (exported != 0).any(axis=-1)
                else:
                    exported = exported != 0
                if exported.shape != occupancy.shape:
                    raise ValueError('Exported PNG dimensions differ')
                union = occupancy > 0
                denominator = int((union | exported).sum())
                score = float((union & exported).sum() / denominator) if denominator else 1.0
                png_iou.append(score)
                image['polygon_vs_png_union_iou'] = score
            images.append(image)
            save_json(out / f'instances/{image_id:04d}.json', dict(image=image, annotations=instances))
            image['instances_sha256'] = sha256(out / f'instances/{image_id:04d}.json')
            if image_id <= 6 and not cached:
                overlay = pixels.copy()
                overlay[occupancy > 0] = (overlay[occupancy > 0]*0.65 + np.array([0, 210, 190])*0.35).astype(np.uint8)
                preview = Image.fromarray(overlay)
                draw = ImageDraw.Draw(preview)
                for obj in instances:
                    box = obj['box_xyxy']
                    draw.rectangle(box, outline=(255, 185, 45), width=2)
                    draw.text((box[0], max(0, box[1]-13)), str(obj['category_id']), fill=(255, 210, 80))
                preview.thumbnail((1200, 600))
                (out / 'previews').mkdir(exist_ok=True)
                preview.save(out / f'previews/{image_id:04d}.png')
            if image_id % 50 == 0:
                print(f'Prepared {image_id}/{len(annotation_paths)} images, {len(annotations)} instances', flush=True)
        except Exception as exc:
            failures.append({'image_id': image_id, 'annotation': str(ann_path), 'error': str(exc)})
    # Group exact decoded-pixel duplicates before the split. No patient IDs are supplied.
    valid_ids = {im['id'] for im in images if im['evaluable']}
    ordered_groups = sorted([g for g in groups if any(i in valid_ids for i in groups[g])],
                            key=lambda g: hashlib.sha256(('20260909:'+g).encode()).hexdigest())
    dev_ids = set()
    for group in ordered_groups:
        if len(dev_ids) >= 60:
            break
        dev_ids.update(groups[group])
    splits = {'seed': 20260909, 'unit': 'decoded_pixel_hash_group',
              'dev': [im['id'] for im in images if im['id'] in dev_ids and im['evaluable']],
              'test': [im['id'] for im in images if im['id'] not in dev_ids and im['evaluable']],
              'excluded': [im['id'] for im in images if not im['evaluable']]}
    for im in images:
        im['split'] = ('dev' if im['id'] in dev_ids else 'test') if im['evaluable'] else 'excluded'
    for row in rows:
        row['split'] = 'dev' if row['image_id'] in dev_ids else 'test'
    audit = {'images': len(images), 'instances': len(annotations), 'expected_images': 598,
             'source_instances': raw_objects, 'expected_source_instances': 15318,
             'evaluable_images':len(valid_ids), 'unique_evaluable_image_groups':len(ordered_groups),
             'excluded_zero_area_objects':invalid_objects,'unannotated_images':empty_images,
             'class_counts': dict(sorted(class_counts.items())),
             'duplicate_pixel_groups': [ids for ids in groups.values() if len(ids)>1],
             'overlap_images': sum(im['overlap_pixels']>0 for im in images),
             'png_comparison_images': len(png_iou),
             'polygon_vs_png_union_iou_mean': float(np.mean(png_iou)) if png_iou else None,
             'polygon_vs_png_union_iou_min': min(png_iou) if png_iou else None,
             'warnings': warnings, 'failures': failures,
             'rasterizer': 'pycocotools COCO polygons, subtract interior rings',
             'bbox_convention': 'XYXY pixel edges, x2/y2 exclusive; COCO export is XYWH',
             'overlap_policy': 'independent RLE per annotation, no overwrite',
             'patient_ids_available': False}
    save_json(out/'audit.json', audit)
    save_json(out/'exclusions.json', {'zero_area_objects':invalid_objects,'unannotated_images':empty_images})
    if failures or raw_objects != 15318 or len(annotations) != 15313 or len(invalid_objects) != 5 or len(empty_images) != 3:
        raise RuntimeError('Dataset validation failed; inspect prepared/audit.json')
    coco = {'info': {'description': 'HITL OPG teeth: canonical COCO polygon rasterization',
                     'source': 'Kaggle dataset version 1', 'box_source': 'ground_truth_mask'},
            'images': images, 'annotations': annotations,
            'categories': [{'id': i, 'name': str(i)} for i in range(1,33)]}
    save_json(out/'coco.json', coco)
    save_json(out/'manifest.json', {'images': images, 'rasterizer': audit['rasterizer']})
    save_json(out/'splits.json', splits)
    with (out/'boxes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['warnings','class_counts']}), flush=True)


if __name__ == '__main__':
    main()
