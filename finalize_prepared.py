"""Validate all saved instances and finalize content hashes before any inference."""
import json
from pathlib import Path
from collections import Counter
from pycocotools import mask as mask_utils
from common import sha256,save_json

root=Path(__file__).resolve().parent
manifest=json.loads((root/'prepared/manifest.json').read_text())
coco=json.loads((root/'prepared/coco.json').read_text())
counts=Counter()
for im in manifest['images']:
    path=root/f'prepared/instances/{im["id"]:04d}.json'
    data=json.loads(path.read_text())
    if len(data['annotations'])!=im['tooth_count']:
        raise ValueError('Tooth count mismatch')
    for ann in data['annotations']:
        box=mask_utils.toBbox(ann['segmentation']).tolist()
        xyxy=[box[0],box[1],box[0]+box[2],box[1]+box[3]]
        if xyxy!=ann['box_xyxy'] or int(mask_utils.area(ann['segmentation']))!=ann['area']:
            raise ValueError('RLE/box/area mismatch')
        if ann['image_id']!=im['id']:
            raise ValueError('Image/instance mismatch')
        counts[ann['id']]+=1
    im['instances_sha256']=sha256(path)
assert len(manifest['images'])==598 and len(counts)==15313 and max(counts.values())==1
coco['images']=manifest['images']
save_json(root/'prepared/manifest.json',manifest)
save_json(root/'prepared/coco.json',coco)
save_json(root/'prepared/validation.json',{'images':598,'unique_instances':15313,
          'all_rle_areas_and_boxes_verified':True,'manifest_sha256':sha256(root/'prepared/manifest.json')})
print('Validated 598 retained images / 15313 valid instances: RLE masks, areas, boxes and content hashes.',flush=True)
