"""Count encoder calls and compare outputs through the real AMG/worker paths."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from PIL import Image
from common import encode
from teethbench.cli import worker_schedule
from teethbench.config import digest, read, write
from teethbench.embeddings import ImageEmbeddingCache
from teethbench.sam3 import automatic_generator
from teethbench.worker import infer_auto, infer_bbox, run_job
from test_sam3 import predictor  # Synthetic SAM 3 fixture; no weights/network needed.


def inputs():
    rgb=np.zeros((40,80,3),dtype=np.uint8)
    mask=np.zeros((40,80),dtype=bool)
    mask[10:30,20:60]=True
    anns=[{'id':i+1,'category_id':1,'box_xyxy':[20,10,60,30],
           'area':int(mask.sum()),'segmentation':encode(mask),'bbox':[20,10,40,20]} for i in range(30)]
    im={'id':1,'width':80,'height':40,'tooth_count':len(anns)}
    plan={'box_condition':'exact','seed':123,'matching_iou':0.5,
          'auto':{'points_per_side':2,'points_per_batch':2,'pred_iou_thresh':0.88,
                  'stability_score_thresh':0.0,'crop_n_layers':0,'output_mode':'coco_rle'}}
    return rgb,anns,im,plan


def predictions_without_timing(result):
    predictions=copy.deepcopy(result['predictions'])
    for prediction in predictions:
        prediction.pop('decode_seconds',None)
    return predictions


@pytest.mark.parametrize('crop_layers,expected_encodes',[(0,1),(1,5)])
def test_thirty_boxes_and_all_grid_batches_share_encoding(predictor,crop_layers,expected_encodes):
    rgb,anns,im,plan=inputs()
    plan['auto']['crop_n_layers']=crop_layers
    cache=ImageEmbeddingCache(predictor,lambda:None)
    generator=automatic_generator(cache,plan['auto'])
    cache.begin_image()
    boxes=infer_bbox(cache,rgb,anns,im,plan,lambda:None)
    auto=infer_auto(cache,generator,rgb,anns,im,plan,lambda:None)
    assert len(boxes['predictions'])==30
    assert predictor.model.encodings==expected_encodes
    assert boxes['encoder_calls']==1 and boxes['embedding_reuses']==0
    assert auto['encoder_calls']==expected_encodes-1 and auto['embedding_reuses']==1
    assert auto['shared_encode_seconds']==boxes['encode_seconds']
    assert auto['inference_seconds']==pytest.approx(auto['actual_inference_seconds']+boxes['encode_seconds'])
    assert predictor._features is None
    # Independent mode execution must produce identical masks, qualities and scores.
    cache.begin_image()
    separate_auto=infer_auto(cache,generator,rgb,anns,im,plan,lambda:None)
    cache.begin_image()
    separate_boxes=infer_bbox(cache,rgb,anns,im,plan,lambda:None)
    for shared,separate in [(boxes,separate_boxes),(auto,separate_auto)]:
        assert predictions_without_timing(shared)==predictions_without_timing(separate)
        assert shared['summary']==separate['summary'] and shared['gt_scores']==separate['gt_scores']
    assert separate_auto['encoder_calls']==expected_encodes and separate_auto['embedding_reuses']==0


def test_cache_rejects_copies_crops_flips_and_previous_images(predictor):
    rgb,_,_,_=inputs()
    cache=ImageEmbeddingCache(predictor,lambda:None)
    cache.set_image(rgb)
    cache.set_image(rgb[:,:,:])
    assert predictor.model.encodings==1
    cache.set_image(rgb.copy())
    cache.set_image(rgb[:20,:40])
    cache.set_image(rgb[:,::-1])
    assert predictor.model.encodings==4
    cache.begin_image()
    rgb[:]=255  # Reusing a buffer for the next image must never reuse stale features.
    cache.set_image(rgb)
    assert predictor.model.encodings==5 and len(cache.events)==1
    cache.reset_predictor()
    assert cache._image is None and predictor._features is None


def test_sam1_reset_interface_and_failed_encoding_clears_cache():
    calls=[]
    def encode_image(image):
        calls.append('encode')
        if image[0,0,0]:
            raise RuntimeError('simulated encoder failure')
    native=SimpleNamespace(reset_image=lambda:calls.append('reset'),set_image=encode_image)
    cache=ImageEmbeddingCache(native,lambda:None)
    image=np.zeros((20,40,3),dtype=np.uint8)
    cache.set_image(image)
    cache.set_image(image[:])
    assert calls.count('encode')==1
    with pytest.raises(RuntimeError):
        cache.set_image(np.ones_like(image))
    assert cache._image is None
    cache.reset_image()
    cache.set_image(image)
    assert calls.count('encode')==3


def test_schedule_loads_each_model_once_for_both_modes():
    plan={'models':['sam1_vit_b','sam2_tiny','sam21_tiny','sam3'],'modes':['auto','bbox']}
    assert list(worker_schedule(plan))==[(m,'both',['bbox','auto']) for m in plan['models']]
    plan['modes']=['auto']
    assert list(worker_schedule(plan))==[(m,'auto',['auto']) for m in plan['models']]


def test_worker_resume_retries_only_missing_mode_and_skips_completed_model(tmp_path,monkeypatch,predictor):
    from teethbench import worker
    rgb,anns,im,plan=inputs()
    image_path=tmp_path/'image.png'
    Image.fromarray(rgb).save(image_path)
    ann_path=tmp_path/'prepared/instances/0001.json'
    write(ann_path,{'annotations':anns})
    (tmp_path/'checkpoints').mkdir()
    checkpoint=tmp_path/'checkpoints/fake.pt'
    checkpoint.write_bytes(b'fake')
    im.update(file_name='image.png',image_sha256=digest(image_path),instances_sha256=digest(ann_path),pixel_sha256='unique')
    plan.update(root=str(tmp_path),images=[im],threads=1,device='cpu',models=['sam21_tiny','sam3'],modes=['auto','bbox'],
                checkpoints_sha256={'sam21_tiny':digest(checkpoint),'sam3':digest(checkpoint)},
                schema='teethbench-v2',image_count=1,image_group_count=1,tooth_count=30,
                split='dev',requested_samples=1,bootstrap=20)
    write(tmp_path/'models.json',{'sam21_tiny':{'family':'sam2.1','checkpoint':'fake.pt'},
                                 'sam3':{'family':'sam3','checkpoint':'fake.pt','transformers_version':'5.17.0'}})
    output=tmp_path/'results'
    write(output/'plan.json',plan)
    monkeypatch.setattr(worker,'verify_plan',lambda plan:None)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    loads=[]
    monkeypatch.setattr(worker,'create_predictor',lambda *args:(loads.append(1) or predictor))
    monkeypatch.setattr(worker,'automatic_generator',lambda p,f,s:automatic_generator(p,s))
    real_auto=worker.infer_auto
    def fail_auto(*args):
        raise RuntimeError('simulated interruption after bbox saved')
    monkeypatch.setattr(worker,'infer_auto',fail_auto)
    with pytest.raises(RuntimeError,match='failed image/mode'):
        run_job(output,'sam21_tiny','both')
    box_path=output/'bbox/sam21_tiny/images/0001.json'
    saved=box_path.read_bytes()
    assert read(output/'bbox/sam21_tiny/status.json')['state']=='complete'
    assert read(output/'auto/sam21_tiny/status.json')['state']=='failed'
    assert predictor._features is None
    monkeypatch.setattr(worker,'infer_auto',real_auto)
    before=predictor.model.encodings
    run_job(output,'sam21_tiny','both')
    assert box_path.read_bytes()==saved
    assert predictor.model.encodings==before+2  # Separate warmup plus missing auto only.
    auto=read(output/'auto/sam21_tiny/images/0001.json')
    assert auto['encoder_calls']==1 and auto['embedding_reuses']==0
    assert not (output/'auto/sam21_tiny/errors/0001.json').exists()
    run_job(output,'sam21_tiny','both')
    assert len(loads)==2  # Fully complete model does not load a third time.
    # Mixed SAM 2/SAM 3 runs record the same installed environment and can rank together.
    from teethbench import sam3
    from teethbench.report import report
    monkeypatch.setattr(sam3,'Sam3Predictor',lambda *args:predictor)
    run_job(output,'sam3','both')
    assert read(output/'auto/sam3/runtime.json')==read(output/'auto/sam21_tiny/runtime.json')
    report(output)
    assert read(output/'summary.json')['complete']
    assert len(read(output/'summary.json')['leaderboards'])==4
