"""Separate automatic/bbox rankings, complete cohorts only, grouped uncertainty."""
import csv
import fcntl
import itertools
from pathlib import Path
import numpy as np
from .config import SCHEMA,read,write
from .evaluation import coco_ap


def csv_file(path,rows):
    path=Path(path)
    if not rows:
        path.write_text('')
        return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def group_values(values,groups):
    return np.array([np.mean([values[i] for i,g in enumerate(groups) if g==group]) for group in sorted(set(groups))])


def interval(values,draws):
    return [float(n) for n in np.quantile(np.asarray(values)[draws].mean(axis=1),[0.025,0.975])]


def report(output):
    output=Path(output)
    plan=read(output/'plan.json')
    if plan.get('schema')!=SCHEMA:
        raise ValueError('Use aggregate.py to report legacy experiments.')
    with (output/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _report(output,plan)


def _report(output,plan):
    groups=[im['pixel_sha256'] for im in plan['images']]
    ng=len(set(groups))
    draws=np.random.default_rng(plan['seed']).integers(0,ng,size=(plan['bootstrap'],ng))
    all_rows,coverage,paired=[],[],[]
    runtime_reference=None
    lines=['# Dental SAM benchmark','',
           f'{plan["image_count"]} images / {ng} independent decoded-image groups / {plan["tooth_count"]} teeth. '
           f'Split: {plan["split"]}. Sample seed: {plan["seed"]}. Device: {plan["device"]}, FP32.','']
    if plan['split']!='test' or plan['requested_samples']:
        lines+=['**Subset/development experiment: these results do not establish the full held-out ranking.**','']
    if ng<20:
        lines+=['**Very small sample: bootstrap intervals are descriptive and unreliable for general model-ranking claims.**','']
    for mode in plan['modes']:
        mode_dir=output/mode
        mode_dir.mkdir(exist_ok=True)
        primary='instance_quality' if mode=='auto' else 'gt_macro_dice'
        rows,image_rows,tooth_rows=[],[],[]
        vectors={}
        for model in plan['models']:
            job=mode_dir/model
            results=[]
            for im in plan['images']:
                path=job/f'images/{im["id"]:04d}.json'
                if path.exists():
                    result=read(path)
                    if result['image_id']!=im['id'] or result['summary']['gt_count']!=im['tooth_count']:
                        raise ValueError(f'Image identity/count mismatch in {path}')
                    results.append(result)
            complete=len(results)==plan['image_count']
            coverage.append({'mode':mode,'model':model,'completed_images':len(results),'expected_images':plan['image_count'],
                             'status':'complete' if complete else 'incomplete' if results else 'pending'})
            if not complete:
                continue
            runtime=read(job/'runtime.json')
            if runtime_reference is None:
                runtime_reference=runtime
            if runtime!=runtime_reference:
                raise ValueError('Cannot compare jobs from different hardware/runtime environments.')
            def grouped(key):
                return group_values([r['summary'][key] for r in results],groups)
            primary_values=grouped(primary)
            vectors[model]=primary_values
            low,high=interval(primary_values,draws)
            state=read(job/'status.json') if (job/'status.json').exists() else {}
            info=read(job/'model_info.json') if (job/'model_info.json').exists() else {}
            row={'mode':mode,'rank':0,'model':model,'images':len(results),'image_groups':ng,
                 'teeth':plan['tooth_count'],'primary_metric':primary,'primary_score':float(primary_values.mean()),
                 'primary_ci_low':low,'primary_ci_high':high,
                 **{key:float(grouped(key).mean()) for key in ['gt_macro_dice','gt_macro_iou','gt_macro_boundary_f1']},
                 'seconds_per_image_mean':float(np.mean([r['inference_seconds'] for r in results])),
                 'seconds_per_image_median':float(np.median([r['inference_seconds'] for r in results])),
                 'seconds_per_image_p95':float(np.percentile([r['inference_seconds'] for r in results],95)),
                 'parameters':info.get('parameters'),'peak_process_rss_gib':state.get('peak_process_rss_gib'),
                 'peak_cuda_allocated_gib':state.get('peak_cuda_allocated_gib')}
            if mode=='auto':
                row.update({key:float(grouped(key).mean()) for key in ['detection_precision','detection_recall','detection_f1','instance_dice_quality']})
                row.update({key:sum(r['summary'][key] for r in results) for key in ['tp','fp','fn','prediction_count']})
                # Cache COCO AP for completed immutable predictions; report can be rerun cheaply.
                ap_path=job/'coco_ap.json'
                if not ap_path.exists():
                    ann_records=[read(Path(plan['root'])/f'prepared/instances/{im["id"]:04d}.json') for im in plan['images']]
                    write(ap_path,coco_ap(plan['images'],results,ann_records))
                row.update(read(ap_path))
            else:
                row['empty_predictions']=sum(r['summary']['empty_predictions'] for r in results)
            rows.append(row)
            for result in results:
                image_rows.append({'model':model,'mode':mode,'image_id':result['image_id'],
                                   'inference_seconds':result['inference_seconds'],**result['summary']})
                for gt in result['gt_scores']:
                    tooth_rows.append({'model':model,'mode':mode,'image_id':result['image_id'],**gt})
        rows.sort(key=lambda row:(-row['primary_score'],row['model']))
        for rank,row in enumerate(rows,1):
            row['rank']=rank
        for a,b in itertools.combinations(vectors,2):
            delta=vectors[a]-vectors[b]
            low,high=interval(delta,draws)
            paired.append({'mode':mode,'model_a':a,'model_b':b,'metric':primary,
                           'mean_a_minus_b':float(delta.mean()),'ci_low':low,'ci_high':high,
                           'interval':'95% pointwise paired image-group bootstrap; not multiplicity adjusted'})
        all_rows.extend(rows)
        csv_file(mode_dir/'leaderboard.csv',rows)
        csv_file(mode_dir/'per_image.csv',image_rows)
        csv_file(mode_dir/'per_tooth.csv',tooth_rows)
        class_rows=[]
        for model in plan['models']:
            for cls in sorted({r['category_id'] for r in tooth_rows if r['model']==model}):
                selected=[r for r in tooth_rows if r['model']==model and r['category_id']==cls]
                class_rows.append({'model':model,'tooth_class':cls,'count':len(selected),
                                   **{key:float(np.mean([r[key] for r in selected])) for key in ['dice','iou','boundary_f1']}})
        csv_file(mode_dir/'per_class.csv',class_rows)
        lines+=[f'## {"Automatic masks" if mode=="auto" else "Ground-truth box prompts"}','',
                f'{len(rows)}/{len(plan["models"])} models complete. Incomplete models are excluded from ranking.','']
        if mode=='auto':
            lines+=[f'Primary: instance quality at matching IoU ≥ {plan["matching_iou"]}: '
                    'sum(matched IoU) / (TP + 0.5 FP + 0.5 FN). Matching is one-to-one, maximizing valid match count before summed IoU. '
                    'This is a PQ-style instance score, not formal panoptic quality, because tooth/proposal masks may overlap.','',
                    '| Rank | Model | Instance quality (95% CI) | GT Dice | F1 | FP | FN | COCO AP | AP50 | Seconds/image |',
                    '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for r in rows:
                lines.append(f'| {r["rank"]} | {r["model"]} | {r["primary_score"]:.4f} ({r["primary_ci_low"]:.4f}–{r["primary_ci_high"]:.4f}) | '
                             f'{r["gt_macro_dice"]:.4f} | {r["detection_f1"]:.4f} | {r["fp"]} | {r["fn"]} | '
                             f'{r["coco_ap"]:.4f} | {r["coco_ap50"]:.4f} | {r["seconds_per_image_mean"]:.2f} |')
        else:
            lines+=[f'Box condition: {plan["box_condition"]}. Primary: mean Dice for each tooth, then each image, then each unique image group.','',
                    '| Rank | Model | Dice (95% CI) | IoU | Boundary F1 (2 px) | Seconds/image |',
                    '|---:|---|---:|---:|---:|---:|']
            for r in rows:
                lines.append(f'| {r["rank"]} | {r["model"]} | {r["primary_score"]:.4f} ({r["primary_ci_low"]:.4f}–{r["primary_ci_high"]:.4f}) | '
                             f'{r["gt_macro_iou"]:.4f} | {r["gt_macro_boundary_f1"]:.4f} | {r["seconds_per_image_mean"]:.2f} |')
        lines+=['']
    done=all(row['status']=='complete' for row in coverage)
    if 'sam3' in plan['models']:
        lines+=['SAM 3 uses the visual instance-segmentation (PVS/tracker) head through the pinned '
                'Hugging Face Transformers backend. It receives no text or concept prompts. Automatic SAM 3 masks '
                'use the same SAM 2 point-grid/crop/filter/NMS implementation with the SAM 3 predictor; all mask '
                'logits and quality scores come from SAM 3. Backend versions and checkpoint/config hashes are saved.','']
    lines+=['## Interpretation','',
            'The two modes are separate tasks and have separate rankings. Bbox mode receives privileged ground-truth location information. '
            'Automatic mode generates general object proposals: every returned proposal is treated as a tooth prediction; extra structures and duplicate masks count as false positives. '
            'There is no tooth classifier, GT-based proposal filtering, GT-based top-k selection, or GT-derived automatic grid.', '',
            'In automatic GT-macro Dice/IoU/boundary scores, unmatched teeth score zero; matched-only diagnostics do not determine the ranking. '
            'The instance-quality denominator additionally penalizes false positives. COCO AP uses the official 0.50:0.05:0.95 thresholds, one tooth class, '
            'model-predicted IoU as confidence, and the standard maximum of 100 detections per image. Instance-quality counts include every returned mask; '
            'COCO AP is the conventional image-weighted statistic, while primary scores and confidence intervals balance identical-image annotation groups.', '',
            f'Confidence intervals use {plan["bootstrap"]} bootstrap replicates over decoded-image groups. Paired differences are reported within each mode only; '
            'intervals are pointwise, not simultaneous. No patient-disjointness or pretraining-data exclusion can be established. '
            'Zero-area annotations and unannotated images follow prepared/exclusions.json.', '',
            'Timing follows native predictor/generator API calls after a warmup. Automatic timing includes point-grid decoding, crop processing, filtering/NMS and native RLE output; '
            'bbox timing includes native encoding and one decode per box. Data reads, metrics and file writes are excluded. Shared CPU load affects timing.','']
    csv_file(output/'comparison.csv',all_rows)
    csv_file(output/'coverage.csv',coverage)
    csv_file(output/'paired_differences.csv',paired)
    write(output/'summary.json',{'schema':SCHEMA,'complete':done,'coverage':coverage,'leaderboards':all_rows,
                                 'paired_differences':paired,'bootstrap_unit':'decoded_pixel_image_group'})
    (output/'REPORT.md').write_text('\n'.join(lines))
