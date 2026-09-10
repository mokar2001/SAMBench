import json
from pathlib import Path
import subprocess
import sys
from common import save_json


def make_run(suite, name, values, condition='exact'):
    meta={'image_ids':[1,2], 'image_groups':['a','b'], 'expected_instances':2, 'manifest_sha256':'dataset',
          'splits_sha256':'splits', 'condition':condition,'seed':1,'split':'test',
          'precision':'float32','multimask_output':False,'postprocessing':False,
          'threshold_logits':0,'boundary_tolerance_px':2,'source_sha256':{'run':'code'},
          'torch':'test','numpy':'test','device':'cpu','threads':2,'host':'test'}
    save_json(suite/name/'run.json',meta)
    for i,dice in enumerate(values,1):
        record={'instance_id':i,'category_id':1,'dice':dice,'iou':dice/(2-dice),
                'boundary_f1':dice,'precision':dice,'recall':dice,'intersection':int(100*dice),
                'gt_area':100,'pred_area':100,'hd95_px':0 if dice==1 else 3,
                'assd_px':0 if dice==1 else 2,'empty':0,'decode_seconds':0.1}
        save_json(suite/name/f'images/{i:04d}.json',{'image_id':i,'instances':[record],
                  'inference_seconds':1,'encode_seconds':0.9})


def run_aggregate(suite):
    return subprocess.run([sys.executable,str(Path(__file__).with_name('aggregate.py')),
                           '--suite',str(suite),'--bootstrap','100'],capture_output=True,text=True)


def test_incomplete_models_cannot_win(tmp_path):
    save_json(tmp_path/'suite.json',{'models':['a','b','c'],'split':'test','condition':'exact'})
    make_run(tmp_path,'a',[0.6,0.8])
    make_run(tmp_path,'b',[1.0])
    result=run_aggregate(tmp_path)
    assert result.returncode==0,result.stderr
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert not summary['complete']
    assert [r['model'] for r in summary['leaderboard']]==['a']
    assert summary['coverage'][1]['status']=='incomplete'
    assert summary['coverage'][2]['status']=='not_started'


def test_paired_difference_and_protocol_guard(tmp_path):
    save_json(tmp_path/'suite.json',{'models':['a','b'],'split':'test','condition':'exact'})
    make_run(tmp_path,'a',[0.8,0.8])
    make_run(tmp_path,'b',[0.6,0.6])
    result=run_aggregate(tmp_path)
    assert result.returncode==0,result.stderr
    summary=json.loads((tmp_path/'summary.json').read_text())
    pair=summary['paired_differences'][0]
    assert abs(pair['mean_image_group_dice_a_minus_b']-0.2)<1e-9
    assert abs(pair['ci_low']-0.2)<1e-9 and abs(pair['ci_high']-0.2)<1e-9
    make_run(tmp_path,'b',[0.6,0.6],condition='pad5')
    result=run_aggregate(tmp_path)
    assert result.returncode!=0 and 'Incompatible protocol' in result.stderr


def test_duplicate_images_are_one_bootstrap_group(tmp_path):
    save_json(tmp_path/'suite.json',{'models':['a'],'split':'test','condition':'exact'})
    make_run(tmp_path,'a',[0.2,0.8])
    path=tmp_path/'a/run.json'
    meta=json.loads(path.read_text())
    meta['image_groups']=['same_pixels','same_pixels']
    save_json(path,meta)
    result=run_aggregate(tmp_path)
    assert result.returncode==0,result.stderr
    row=json.loads((tmp_path/'summary.json').read_text())['leaderboard'][0]
    assert row['images']==2 and row['image_groups']==1
    assert row['image_group_macro_dice']==0.5
    assert row['dice_ci_low']==row['dice_ci_high']==0.5
