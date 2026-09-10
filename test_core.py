import numpy as np
import pytest
from common import box_from_mask, decode, encode, metrics, polygon_mask, prompt_box


def test_perfect_and_empty_predictions():
    gt = np.zeros((30,40), dtype=bool)
    gt[0:10,0:10] = True
    exact = metrics(gt,gt)
    assert exact['dice']==exact['iou']==exact['boundary_f1']==1.0
    assert exact['hd95_px']==exact['assd_px']==0
    empty = metrics(np.zeros_like(gt),gt)
    assert empty['dice']==empty['iou']==empty['boundary_f1']==0
    assert empty['hd95_px'] is None and empty['empty']==1


def test_known_shifted_rectangles():
    gt = np.zeros((30,40), dtype=bool)
    gt[10:20,10:20] = True
    pred = np.zeros_like(gt)
    pred[10:20,12:22] = True
    score = metrics(pred,gt)
    assert score['dice']==pytest.approx(0.8)
    assert score['iou']==pytest.approx(2/3)
    assert score['hd95_px']==pytest.approx(2)
    assert score['boundary_f1']==1


def test_disjoint_masks_and_wrong_shape():
    gt = np.eye(10,dtype=bool)
    pred = np.fliplr(gt)
    assert metrics(pred,gt)['dice']==0
    with pytest.raises(ValueError):
        metrics(pred[:5],gt)


def test_holes_rle_boxes_and_overlap():
    poly = {'exterior': [[2,2],[12,2],[12,12],[2,12]],
            'interior': [[[5,5],[9,5],[9,9],[5,9]]]}
    mask = polygon_mask(poly,20,20)
    assert not mask[6,6] and mask[3,3]
    assert np.array_equal(decode(encode(mask)),mask)
    assert box_from_mask(mask)==[2,2,12,12]
    other = polygon_mask({'exterior':[[8,8],[18,8],[18,18],[8,18]]},20,20)
    # Two RLE records must preserve the overlap, unlike a single instance-label PNG.
    assert (decode(encode(mask)) & decode(encode(other))).any()


def test_prompt_identity_jitter_and_clipping():
    box = [0,0,20,10]
    assert np.array_equal(prompt_box(box,30,30,'exact',1,123),box)
    assert np.array_equal(prompt_box(box,30,30,'pad10',1,123),[0,0,22,11])
    a = prompt_box(box,30,30,'jitter5',12,123)
    b = prompt_box(box,30,30,'jitter5',12,123)
    assert np.array_equal(a,b)
    assert not np.array_equal(a,prompt_box(box,30,30,'jitter5',13,123))
    one = np.zeros((10,10),dtype=bool)
    one[-1,-1]=True
    assert box_from_mask(one)==[9,9,10,10]
