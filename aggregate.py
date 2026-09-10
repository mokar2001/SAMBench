"""Rank only complete, protocol-matched runs; bootstrap whole radiographs."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from common import save_json


def write_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def ci(values, draws):
    means = np.asarray(values, dtype=float)[draws].mean(axis=1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=5000)
    args = p.parse_args()
    suite = args.suite.resolve()
    suite_spec = json.loads((suite/'suite.json').read_text())
    expected = suite_spec['models']
    complete, coverage = {}, []
    signature = None
    compare_keys = ['image_ids','image_groups','manifest_sha256','splits_sha256','condition','seed','split',
                    'precision','multimask_output','postprocessing','threshold_logits','boundary_tolerance_px',
                    'source_sha256','torch','numpy','device','threads','host']
    for name in expected:
        folder = suite/name
        if not (folder/'run.json').exists():
            coverage.append({'model':name, 'status':'not_started', 'completed_images':0, 'expected_images':None})
            continue
        meta = json.loads((folder/'run.json').read_text())
        sig = {k:meta[k] for k in compare_keys}
        if signature is None:
            signature = sig
        if sig != signature:
            raise ValueError(f'Incompatible protocol for {name}; create a separate suite')
        rows = []
        for image_id in meta['image_ids']:
            path = folder/f'images/{image_id:04d}.json'
            if path.exists():
                rows.append(json.loads(path.read_text()))
        done = len(rows)==len(meta['image_ids']) and sum(len(r['instances']) for r in rows)==meta['expected_instances']
        coverage.append({'model':name, 'status':'complete' if done else 'incomplete',
                         'completed_images':len(rows), 'expected_images':len(meta['image_ids'])})
        if done:
            complete[name] = (meta,rows)
    suite.mkdir(parents=True, exist_ok=True)
    write_csv(suite/'coverage.csv', coverage)
    leaderboard, per_image, per_tooth, per_class, paired = [], [], [], [], []
    vectors = {}
    if complete:
        n = len(next(iter(complete.values()))[1])
        hashes = next(iter(complete.values()))[0]['image_groups']
        group_indices = [[i for i,g in enumerate(hashes) if g==group] for group in sorted(set(hashes))]
        ng = len(group_indices)
        draws = np.random.default_rng(20260909).integers(0,ng,size=(args.bootstrap,ng))
        for name,(meta,images) in complete.items():
            status_path = suite/name/'status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            teeth = []
            image_stats = []
            for im in images:
                stats = {'model':name,'image_id':im['image_id'],'teeth':len(im['instances']),
                         'inference_seconds':im['inference_seconds'],'encode_seconds':im['encode_seconds']}
                for metric in ['dice','iou','boundary_f1','precision','recall']:
                    stats[metric] = float(np.mean([r[metric] for r in im['instances']]))
                stats['failure_iou_lt_0_5'] = float(np.mean([r['iou']<0.5 for r in im['instances']]))
                image_stats.append(stats)
                per_image.append(stats)
                for inst in im['instances']:
                    row = {'model':name,'image_id':im['image_id'],
                           **{k:v for k,v in inst.items() if k not in ['segmentation','prompt_box_xyxy']}}
                    teeth.append(row)
                    per_tooth.append(row)
            def group_scores(key):
                return np.array([np.mean([image_stats[i][key] for i in indices]) for indices in group_indices])
            dice = group_scores('dice')
            iou = group_scores('iou')
            vectors[name] = dice
            dice_ci, iou_ci = ci(dice,draws), ci(iou,draws)
            intersection = sum(r['intersection'] for r in teeth)
            area_sum = sum(r['gt_area']+r['pred_area'] for r in teeth)
            distances = [r['hd95_px'] for r in teeth if r['hd95_px'] is not None]
            leaderboard.append({'rank':0, 'model':name,'images':n,'image_groups':ng,'teeth':len(teeth),
                                'parameters':status.get('parameter_count'),
                                'peak_process_rss_gib':status.get('peak_process_rss_gib'),
                                'peak_cuda_allocated_gib':status.get('peak_cuda_allocated_gib'),
                                'image_group_macro_dice':float(dice.mean()),'dice_ci_low':dice_ci[0],'dice_ci_high':dice_ci[1],
                                'image_group_macro_iou':float(iou.mean()),'iou_ci_low':iou_ci[0],'iou_ci_high':iou_ci[1],
                                'image_group_macro_boundary_f1_2px':float(group_scores('boundary_f1').mean()),
                                'instance_macro_dice':float(np.mean([r['dice'] for r in teeth])),
                                'micro_dice':2*intersection/area_sum,
                                'micro_iou':intersection/(area_sum-intersection),
                                'image_group_macro_failure_iou_lt_0_5':float(group_scores('failure_iou_lt_0_5').mean()),
                                'empty_predictions':sum(r['empty'] for r in teeth),
                                'hd95_px_mean_nonempty_only':float(np.mean(distances)) if distances else None,
                                'undefined_hd95_count':len(teeth)-len(distances),
                                'seconds_per_image_mean':float(np.mean([r['inference_seconds'] for r in images])),
                                'seconds_per_image_median':float(np.median([r['inference_seconds'] for r in images])),
                                'seconds_per_image_p95':float(np.percentile([r['inference_seconds'] for r in images],95))})
            for cls in sorted({r['category_id'] for r in teeth}):
                selected = [r for r in teeth if r['category_id']==cls]
                per_class.append({'model':name,'tooth_class':cls,'instances':len(selected),
                                  **{k:float(np.mean([r[k] for r in selected])) for k in ['dice','iou','boundary_f1']}})
        leaderboard.sort(key=lambda r:(-r['image_group_macro_dice'],r['model']))
        for i,row in enumerate(leaderboard,1):
            row['rank']=i
        for a,b in itertools.combinations(vectors,2):
            delta = vectors[a]-vectors[b]
            interval = ci(delta,draws)
            paired.append({'model_a':a,'model_b':b,'mean_image_group_dice_a_minus_b':float(delta.mean()),
                           'ci_low':interval[0],'ci_high':interval[1],
                           'note':'paired decoded-pixel image-group bootstrap; 95% pointwise CI, not adjusted for multiple comparisons'})
    for filename,rows in [('leaderboard.csv',leaderboard),('per_image.csv',per_image),('per_tooth.csv',per_tooth),
                          ('per_class.csv',per_class),('paired_differences.csv',paired)]:
        write_csv(suite/filename,rows)
    finished = len(complete)==len(expected)
    summary = {'complete':finished, 'completed_models':len(complete),'expected_models':len(expected),
               'coverage':coverage,'bootstrap_replicates':args.bootstrap,'bootstrap_unit':'decoded_pixel_image_group',
               'leaderboard':leaderboard,'paired_differences':paired}
    save_json(suite/'summary.json',summary)
    lines = ['# OPG box-prompt benchmark', '',
             f'Status: {len(complete)}/{len(expected)} models complete. ' + ('Final for the specified suite.' if finished else 'Partial suite; no overall winner can yet be declared.'), '',
             f'Protocol: {suite_spec.get("split")}; {suite_spec.get("condition")}; FP32; one independent box per tooth.', '',
             'Primary score: average tooth Dice within each image; average duplicate-image annotation sets within each decoded-pixel image group; average over groups. Confidence intervals resample whole groups (5,000 draws by default).', '',
             '| Rank | Model | Images | Teeth | Dice (95% CI) | IoU | Boundary F1 (2 px) | Seconds/image |',
             '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in leaderboard:
        lines.append(f'| {r["rank"]} | {r["model"]} | {r["images"]} | {r["teeth"]} | '
                     f'{r["image_group_macro_dice"]:.4f} ({r["dice_ci_low"]:.4f}–{r["dice_ci_high"]:.4f}) | '
                     f'{r["image_group_macro_iou"]:.4f} | {r["image_group_macro_boundary_f1_2px"]:.4f} | {r["seconds_per_image_mean"]:.2f} |')
    lines += ['', 'These are ground-truth-box-assisted results, not detection or fully automatic segmentation. '
              'Ranks order point estimates; paired_differences.csv reports uncertainty in differences. '
              'Intervals are pointwise, not simultaneous. No millimeter claims are made because pixel spacing is unavailable. '
              'HD95 is descriptive for nonempty predictions only; inspect undefined counts and empty-mask failures. '
              'This single dataset cannot establish universal clinical superiority.', '',
              'Runtime includes native image preprocessing/encoding and per-box decoding, excludes loading, metric calculation and disk writes. '
              'Shared-server timing may vary with other workloads. CPU/GPU runs must use separate suites.', '']
    if suite_spec.get('split') == 'dev':
        lines[2:2] = ['**Development check only. This is not the held-out benchmark and cannot establish a best model.**', '']
    (suite/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'complete':finished,'completed_models':len(complete),'expected_models':len(expected)}))


if __name__ == '__main__':
    main()
