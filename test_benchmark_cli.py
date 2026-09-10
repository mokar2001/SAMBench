import json
from pathlib import Path
import numpy as np
import pytest
from common import encode
from teethbench.cli import parser
from teethbench.config import FAMILIES,sample_images,select_models
from teethbench.evaluation import coco_ap,evaluate_auto,match_instances
from teethbench.report import group_values,report


def annotation(mask,identifier=1):
    y,x=np.nonzero(mask)
    return {'id':identifier,'category_id':identifier,'area':int(mask.sum()),'segmentation':encode(mask),
            'bbox':[int(x.min()),int(y.min()),int(x.max()-x.min()+1),int(y.max()-y.min()+1)]}


def prediction(mask,score=0.9):
    return {'segmentation':encode(mask),'predicted_quality':score}


def shapes():
    a=np.zeros((40,60),dtype=bool)
    b=a.copy()
    extra=a.copy()
    a[5:15,5:15]=True
    b[5:15,25:35]=True
    extra[25:35,45:55]=True
    return a,b,extra


def test_match_maximizes_valid_cardinality_before_overlap():
    matrix=np.array([[0.9,0.51],[0.5,0.49]])
    assert set(match_instances(matrix,0.5))=={(0,1),(1,0)}
    assert match_instances(np.zeros((0,3)),0.5)==[]
    assert match_instances(np.array([[0.5]]),0.5)==[(0,0)]
    with pytest.raises(ValueError):
        match_instances(matrix,0)


def test_auto_perfect_masks_are_order_independent():
    a,b,_=shapes()
    result=evaluate_auto([prediction(b),prediction(a)],[annotation(a),annotation(b,2)])
    score=result['summary']
    assert score['tp']==2 and score['fp']==score['fn']==0
    assert score['instance_quality']==score['gt_macro_dice']==score['detection_f1']==1
    assert result['matches'][0]['instance_id']==2


def test_auto_penalizes_duplicate_and_non_tooth_proposals():
    a,_,extra=shapes()
    result=evaluate_auto([prediction(a),prediction(a),prediction(extra)],[annotation(a)])
    s=result['summary']
    assert s['tp']==1 and s['fp']==2 and s['fn']==0
    assert s['instance_quality']==0.5 and s['detection_precision']==pytest.approx(1/3)
    assert s['gt_macro_dice']==1  # This diagnostic alone would hide the false positives.
    assert len(result['false_positive_indices'])==2


def test_auto_missed_teeth_and_empty_predictions_score_zero():
    a,b,extra=shapes()
    gt=[annotation(a),annotation(b,2)]
    result=evaluate_auto([prediction(a),prediction(extra)],gt)
    s=result['summary']
    assert (s['tp'],s['fp'],s['fn'])==(1,1,1)
    assert s['gt_macro_dice']==s['instance_quality']==0.5
    empty=evaluate_auto([],gt)['summary']
    assert empty['tp']==empty['fp']==0 and empty['fn']==2
    assert empty['instance_quality']==empty['gt_macro_dice']==0
    assert empty['matched_mean_iou'] is None


def test_coco_ap_perfect_and_no_detections():
    a,b,_=shapes()
    images=[{'id':1,'width':60,'height':40}]
    truth=[{'annotations':[annotation(a),annotation(b,2)]}]
    perfect=coco_ap(images,[{'predictions':[prediction(a),prediction(b,0.8)]}],truth)
    assert perfect['coco_ap']==pytest.approx(1)
    assert perfect['coco_ap50']==pytest.approx(1)
    empty=coco_ap(images,[{'predictions':[]}],truth)
    assert empty['coco_ap']==empty['coco_ar100']==0


def test_model_size_validation_and_mixed_generation_comparison():
    registry={name:{} for family in FAMILIES.values() for name in family.values()}
    assert select_models(registry,family='sam2.1',size='tiny')==['sam21_tiny']
    assert select_models(registry,family='sam1',size='huge')==['sam1_vit_h']
    assert select_models(registry,models=['sam21_tiny','sam21_large','sam1_vit_h'])==['sam21_tiny','sam21_large','sam1_vit_h']
    with pytest.raises(ValueError,match='no huge checkpoint'):
        select_models(registry,family='sam2.1',size='huge')
    with pytest.raises(ValueError,match='Duplicate'):
        select_models(registry,models=['sam21_tiny','sam21_tiny'])
    with pytest.raises(ValueError,match='either'):
        select_models(registry,models=['sam21_tiny'],size='tiny')


def test_sampling_is_seeded_counts_images_and_keeps_duplicates_together():
    images=[{'id':i,'evaluable':True,'split':'test','pixel_sha256':str(i//2 if i<2 else i)} for i in range(15)]
    a=sample_images(images,'test',5,12)
    assert a==sample_images(images,'test',5,12)
    assert len(a)==5 and a!=sample_images(images,'test',5,13)
    ids={im['id'] for im in a}
    assert (0 in ids)==(1 in ids)
    assert len(sample_images(images,'test',0,12))==15
    with pytest.raises(ValueError,match='exceeds'):
        sample_images(images,'test',16,12)
    with pytest.raises(ValueError):
        sample_images(images,'test',-1,12)
    assert np.array_equal(group_values([0,1,1],['same','same','other']),[1,0.5])


@pytest.mark.parametrize('flags',[['--samples','-1'],['--points-per-side','0'],['--threads','0'],['--pred-iou-thresh','2']])
def test_invalid_cli_numeric_values_fail_before_work(flags):
    with pytest.raises(SystemExit) as exc:
        parser().parse_args(['run','--mode','auto','--models','sam21_tiny',*flags])
    assert exc.value.code==2


def test_partial_reports_never_rank_missing_models(tmp_path):
    plan={'schema':'teethbench-v2','models':['sam21_tiny'],'modes':['auto','bbox'],
          'root':str(tmp_path),'images':[{'id':1,'pixel_sha256':'unique','tooth_count':1}],
          'image_count':1,'image_group_count':1,'tooth_count':1,'seed':12,'split':'dev',
          'requested_samples':1,'device':'cpu','bootstrap':20,'matching_iou':0.5,'box_condition':'exact'}
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    report(tmp_path)
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert not summary['complete'] and summary['leaderboards']==[]
    assert len(summary['coverage'])==2
    assert 'Automatic masks' in (tmp_path/'REPORT.md').read_text()
    assert 'Ground-truth box prompts' in (tmp_path/'REPORT.md').read_text()
