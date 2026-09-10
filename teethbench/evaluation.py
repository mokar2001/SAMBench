"""Class-agnostic automatic instance evaluation, with all FP/FN retained."""
import contextlib
import io
import numpy as np
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment
from common import decode,metrics


def match_instances(iou, threshold):
    """Maximize valid match count first, IoU second; predictions x ground truth."""
    iou=np.asarray(iou,dtype=float)
    if iou.ndim!=2 or not np.isfinite(iou).all() or not 0<threshold<=1:
        raise ValueError('Invalid IoU matrix or matching threshold')
    if not iou.size:
        return []
    valid=iou>=threshold
    bonus=min(iou.shape)+1
    rows,cols=linear_sum_assignment(np.where(valid,bonus+iou,0.0),maximize=True)
    return [(int(r),int(c)) for r,c in zip(rows,cols) if valid[r,c]]


def evaluate_auto(predictions, annotations, threshold=0.5):
    if not annotations:
        raise ValueError('Unannotated images are not valid negative examples.')
    iou=mu.iou([p['segmentation'] for p in predictions],
               [a['segmentation'] for a in annotations],[0]*len(annotations)) if predictions else np.zeros((0,len(annotations)))
    pairs=match_instances(iou,threshold)
    by_gt={g:p for p,g in pairs}
    per_gt=[]
    for g,ann in enumerate(annotations):
        gt=decode(ann['segmentation'])
        index=by_gt.get(g)
        score=metrics(decode(predictions[index]['segmentation']) if index is not None else np.zeros_like(gt),gt)
        per_gt.append({'instance_id':ann['id'],'category_id':ann['category_id'],
                       'prediction_index':index,**score})
    tp=len(pairs)
    fp,fn=len(predictions)-tp,len(annotations)-tp
    denominator=tp+0.5*fp+0.5*fn
    iou_sum=sum(float(iou[p,g]) for p,g in pairs)
    dice_sum=sum(row['dice'] for row in per_gt)
    summary={'gt_count':len(annotations),'prediction_count':len(predictions),'tp':tp,'fp':fp,'fn':fn,
             'instance_quality':iou_sum/denominator,
             'instance_dice_quality':dice_sum/denominator,
             'detection_precision':tp/len(predictions) if predictions else 0.0,
             'detection_recall':tp/len(annotations),'detection_f1':tp/denominator,
             'matched_mean_iou':iou_sum/tp if tp else None,
             'matched_mean_dice':dice_sum/tp if tp else None,
             **{f'gt_macro_{key}':float(np.mean([r[key] for r in per_gt])) for key in ['dice','iou','boundary_f1']}}
    return {'summary':summary,'gt_scores':per_gt,
            'matches':[{'prediction_index':p,'instance_id':annotations[g]['id'],'iou':float(iou[p,g])} for p,g in pairs],
            'false_positive_indices':[p for p in range(len(predictions)) if p not in {x[0] for x in pairs}]}


def coco_ap(images, image_results, annotation_records):
    """Official COCO mask AP; predicted quality supplies scores, no GT filtering."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    truth, detections=[],[]
    for im, result, ann_record in zip(images,image_results,annotation_records):
        for ann in ann_record['annotations']:
            truth.append({'id':ann['id'],'image_id':im['id'],'category_id':1,
                          'segmentation':ann['segmentation'],'area':ann['area'],'bbox':ann['bbox'],'iscrowd':0})
        for pred in result['predictions']:
            detections.append({'image_id':im['id'],'category_id':1,'segmentation':pred['segmentation'],
                               'score':float(pred['predicted_quality'])})
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO()
        gt.dataset={'info':{},'images':[{'id':im['id'],'height':im['height'],'width':im['width']} for im in images],
                    'categories':[{'id':1,'name':'tooth'}],'annotations':truth}
        gt.createIndex()
        if detections:
            dt=gt.loadRes(detections)
        else:
            dt=COCO()
            dt.dataset={**gt.dataset,'annotations':[]}
            dt.createIndex()
        evaluator=COCOeval(gt,dt,'segm')
        evaluator.params.imgIds=[im['id'] for im in images]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {'coco_ap':float(evaluator.stats[0]),'coco_ap50':float(evaluator.stats[1]),
            'coco_ap75':float(evaluator.stats[2]),'coco_ar100':float(evaluator.stats[8])}
