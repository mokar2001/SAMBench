import hashlib
import json
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def encode(mask):
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def decode(rle):
    return mask_utils.decode(rle).astype(bool)


def polygon_mask(points, height, width):
    """COCO pixel-center rasterization; each Supervisely object remains separate."""
    def raster(polygon):
        vertices = np.asarray(polygon, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError('Invalid polygon')
        if not np.isfinite(vertices).all():
            raise ValueError('Non-finite polygon coordinates')
        return mask_utils.decode(mask_utils.frPyObjects([vertices.ravel().tolist()], height, width))[:, :, 0].astype(bool)
    result = raster(points['exterior'])
    for hole in points.get('interior', []):
        result &= ~raster(hole)
    if not result.any():
        raise ValueError('Empty polygon mask')
    return result


def box_from_mask(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError('Empty ground truth')
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def prompt_box(box, width, height, condition, instance_id, seed):
    box = np.array(box, dtype=float)
    wh = box[2:] - box[:2]
    if condition.startswith('pad'):
        amount = float(condition[3:]) / 100
        box += np.r_[-wh * amount, wh * amount]
    elif condition.startswith('jitter'):
        amount = float(condition[6:]) / 100
        digest = hashlib.sha256(f'{seed}:{instance_id}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
        box += rng.uniform(-amount, amount, 4) * np.tile(wh, 2)
    elif condition != 'exact':
        raise ValueError(condition)
    box[0::2] = np.clip(box[0::2], 0, width)
    box[1::2] = np.clip(box[1::2], 0, height)
    if np.any(box[2:] <= box[:2]):
        raise ValueError('Degenerate prompt')
    return box.astype(np.float32)


def metrics(pred, gt, tolerance=2.0):
    """Original-resolution foreground metrics. Boundary distances are in pixels."""
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape or not gt.any():
        raise ValueError('Invalid prediction shape or empty ground truth')
    tp = int(np.count_nonzero(pred & gt))
    pa, ga = int(pred.sum()), int(gt.sum())
    result = {'dice': 2 * tp / (pa + ga), 'iou': tp / (pa + ga - tp),
              'precision': tp / pa if pa else 0.0, 'recall': tp / ga,
              'gt_area': ga, 'pred_area': pa, 'intersection': tp, 'empty': int(pa == 0)}
    if not pa:
        return dict(result, boundary_f1=0.0, hd95_px=None, assd_px=None)
    # Tight union crop preserves full-image distances and avoids expensive background EDTs.
    y, x = np.nonzero(pred | gt)
    sl = (slice(max(0, y.min()-1), min(pred.shape[0], y.max()+2)),
          slice(max(0, x.min()-1), min(pred.shape[1], x.max()+2)))
    p, g = pred[sl], gt[sl]
    pb = p ^ ndimage.binary_erosion(p, border_value=0)
    gb = g ^ ndimage.binary_erosion(g, border_value=0)
    p_to_g = ndimage.distance_transform_edt(~gb)[pb]
    g_to_p = ndimage.distance_transform_edt(~pb)[gb]
    bp, br = float((p_to_g <= tolerance).mean()), float((g_to_p <= tolerance).mean())
    result.update(boundary_f1=2*bp*br/(bp+br) if bp+br else 0.0,
                  hd95_px=float(max(np.percentile(p_to_g, 95), np.percentile(g_to_p, 95))),
                  assd_px=float((p_to_g.sum()+g_to_p.sum())/(len(p_to_g)+len(g_to_p))))
    return result
