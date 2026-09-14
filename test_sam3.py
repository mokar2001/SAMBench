"""SAM 3 adapter contracts, independent of gated weights and network access."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from teethbench.config import select_models
from teethbench.sam3 import Sam3Predictor,automatic_generator


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor=torch.nn.Parameter(torch.zeros(1))
        self.encodings=0
        self.last=None

    @property
    def device(self):
        return self.anchor.device

    def get_image_embeddings(self,pixels):
        self.encodings+=1
        return [pixels.mean().reshape(1,1,1,1)]

    def forward(self,**inputs):
        self.last=inputs
        prompts=inputs.get('input_boxes',inputs.get('input_points'))
        count=prompts.shape[1]
        channels=3 if inputs['multimask_output'] else 1
        logits=torch.full((1,count,channels,4,4),-3.0)
        logits[...,1:3,1:3]=3.0
        return SimpleNamespace(pred_masks=logits,iou_scores=torch.full((1,count,channels),0.99))


class FakeProcessor:
    target_size=1008

    def __call__(self,images,return_tensors):
        return {'pixel_values':torch.from_numpy(images.copy()).permute(2,0,1)[None].float()}

    def post_process_masks(self,masks,original_sizes,**kwargs):
        assert kwargs['binarize'] is False
        assert kwargs['max_hole_area']==kwargs['max_sprinkle_area']==0
        assert kwargs['apply_non_overlapping_constraints'] is False
        return [torch.nn.functional.interpolate(masks[0],size=original_sizes[0],mode='bilinear',align_corners=False)]


@pytest.fixture
def predictor():
    from sam2.utils.transforms import SAM2Transforms
    p=Sam3Predictor.__new__(Sam3Predictor)
    p.model=FakeModel()
    p.processor=FakeProcessor()
    p.image_size=p.processor.target_size
    p.mask_threshold=0.0
    p._transforms=SAM2Transforms(p.image_size,mask_threshold=0.0,max_hole_area=0,max_sprinkle_area=0)
    p.reset_predictor()
    return p


def test_sam3_box_coordinates_cache_and_original_resolution(predictor):
    image=np.zeros((40,80,3),dtype=np.uint8)
    predictor.set_image(image)
    masks,scores,low=predictor.predict(box=np.array([10,5,30,15]),multimask_output=False)
    assert masks.shape==(1,40,80) and masks.dtype==bool
    assert scores.shape==(1,) and low.shape==(1,4,4)
    assert predictor.model.last['input_boxes'].tolist()==[[[126,126,378,378]]]
    assert predictor.model.last['multimask_output'] is False
    predictor.predict(box=np.array([0,0,80,40]),multimask_output=False)
    assert predictor.model.encodings==1
    predictor.set_image(np.full_like(image,255))
    assert predictor.model.encodings==2
    predictor.reset_predictor()
    with pytest.raises(ValueError,match='set_image'):
        predictor.predict(box=np.array([0,0,10,10]))


def test_sam3_amg_batched_points_are_not_rescaled_twice(predictor):
    predictor.set_image(np.zeros((40,80,3),dtype=np.uint8))
    points=torch.tensor([[[252.,504.]],[[756.,504.]]])
    masks,scores,low=predictor._predict(points,torch.ones((2,1),dtype=torch.int),return_logits=True)
    assert masks.shape==(2,3,40,80) and scores.shape==(2,3)
    assert low.shape==(2,3,4,4)
    assert torch.equal(predictor.model.last['input_points'],points[None])
    assert masks.min()<0 and masks.max()>0


def test_sam3_automatic_generator_uses_the_sam3_predictor(predictor):
    generator=automatic_generator(predictor,{'points_per_side':2,'points_per_batch':2,
        'pred_iou_thresh':0.88,'stability_score_thresh':0.0,'crop_n_layers':0,'output_mode':'coco_rle'})
    records=generator.generate(np.zeros((40,80,3),dtype=np.uint8))
    assert records and generator.predictor is predictor
    assert predictor.model.encodings==1
    assert predictor._features is None  # AMG releases each crop's cached image.
    assert all(r['segmentation']['size']==[40,80] for r in records)
    assert all(r['predicted_iou']==pytest.approx(0.99) for r in records)


def test_sam3_selection_and_ungated_group():
    registry=json.loads((Path(__file__).parent/'models.json').read_text())
    assert select_models(registry,family='sam3')==['sam3']
    assert len(select_models(registry,models=['all']))==12
    assert len(select_models(registry,models=['ungated']))==11
    assert 'sam3' not in select_models(registry,models=['ungated'])
    with pytest.raises(ValueError,match='no tiny checkpoint'):
        select_models(registry,family='sam3',size='tiny')


@pytest.mark.parametrize('use_symlink',[False,True])
def test_hf_download_pins_revision_and_hashes_config_files(tmp_path,monkeypatch,use_symlink):
    import hashlib
    import huggingface_hub
    from download_checkpoints import fetch_checkpoint
    spec={'family':'sam3','checkpoint':'sam3/model.safetensors','hf_repo':'facebook/sam3',
          'hf_revision':'pinned-revision','hf_files':['config.json','processor_config.json','model.safetensors']}
    root=tmp_path/'project'
    root.mkdir()
    if use_symlink:
        shared=tmp_path/'shared'
        shared.mkdir()
        (root/'checkpoints').symlink_to(shared,target_is_directory=True)
    calls=[]
    def fake_download(repo,filename,revision,local_dir):
        calls.append((repo,filename,revision))
        local_dir.mkdir(parents=True,exist_ok=True)
        path=local_dir/filename
        path.write_bytes(filename.encode())
        return str(path.resolve())
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',fake_download)
    record=fetch_checkpoint(root,'sam3',spec)
    assert [r[2] for r in calls]==['pinned-revision']*3
    assert record['sha256']==hashlib.sha256(b'model.safetensors').hexdigest()
    assert record['files_sha256']['sam3/config.json']==hashlib.sha256(b'config.json').hexdigest()
    assert json.loads((root/'checkpoints/sam3.json').read_text())==record
