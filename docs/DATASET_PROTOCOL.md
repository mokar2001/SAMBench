# Dataset audit and legacy bounding-box protocol

For the current two-mode CLI, see [CLI_GUIDE.md](../CLI_GUIDE.md).
The commands below document the original box-only experiment.

## Data and outputs

- `downloads/teeth.zip`: complete original Kaggle archive; `dataset_provenance.json`
  records its SHA256, version, CRC inventory, source and license.
- `raw/`: all original JSON and PNG exports, extracted with ZIP CRC validation.
- `prepared/boxes.csv`: one row per annotated tooth, original image path and size,
  tooth class, area and exact bounding box. Coordinates are XYXY pixel edges, with
  the upper x/y edges exclusive. Boxes come from nonzero ground-truth mask pixels.
- `prepared/coco.json`: images, independent instance RLE masks, classes, COCO XYWH
  boxes, and source object IDs. Overlapping masks remain independent.
- `prepared/instances/`: per-image copies for efficient inference.
- `prepared/manifest.json`, `splits.json`, `audit.json`: hashes, fixed cohorts and
  all annotation validation findings. `previews/` displays masks and boxes.
- `checkpoints/`: all 11 official SAM checkpoints, each with URL, size and SHA256.
- `repos/`: official repositories. Actual commits are captured in every run.
- `results/<suite>/<model>/images/`: predicted instance RLE masks, prompt boxes,
  model quality scores, measured scores and timing, saved atomically per image.
- `results/<suite>/REPORT.md`, `leaderboard.csv`, `per_image.csv`, `per_tooth.csv`,
  `per_class.csv`, `paired_differences.csv`, `coverage.csv`: reports and raw scores.

The dataset lists **598 images and 15,318 tooth polygons**. Validation found **15,313
evaluable masks across 595 annotated images**: five polygons are zero-area lines or
repeated points, and three images have no annotations. Every excluded object and
image is recorded in `prepared/exclusions.json` with its source identity and reason.
All 598 images remain in the manifest and original data; excluded images have split
`excluded`. The fixed cohorts contain **60 development images / 1,543 teeth** and
**535 test images / 13,770 teeth**. The test images form **534 unique image groups**.
Files `403.jpg` and `404.jpg` have identical decoded pixels but different annotation
sets: both are retained, kept in one split, and balanced as one group in scoring.

The two archive exports
contain duplicate copies of the images; preparation uses only the JSON export's
image/annotation pairs. It audits agreement with non-black pixels in the PNG export.
The canonical scoring masks use pycocotools polygon rasterization, with interior
rings subtracted. The PNG diagnostic is **not a ground-truth equivalence test**:
mean non-black union agreement is 0.9558 and minimum 0.8228. Inspection of `144.jpg`
found tooth class 32 (20,084 polygon-mask pixels) rendered entirely black, like the
background, despite matching polygon geometry in both JSON exports. Consequently,
turning PNG non-black pixels into foreground can silently lose annotated teeth.
The benchmark consistently uses the original JSON polygons and independent RLEs.
No images or objects are silently removed; unexpected counts, previously unknown
empty polygons or invalid dimensions stop preparation. The five verified degenerate
polygons are excluded explicitly; valid objects in their images remain. Repeated class labels within an image and
overlaps are retained and listed in the audit.

## Frozen experiment protocol

This measures **zero-shot, ground-truth-box-assisted tooth instance segmentation**.
It does not evaluate automatic tooth detection, tooth numbering, or a complete
clinical workflow. Tooth labels identify evaluation strata and are never fed to SAM.

1. Reserve 60 annotated images for implementation checks; the remaining annotated images
   form the held-out test cohort. Split deterministically with seed 20260909, grouping
   exact decoded-pixel duplicates together. Patient IDs are not supplied, so this is
   an image split, not a verified patient-disjoint split. Near duplicates and repeated
   patients require further review before strong publication claims.
2. Load original images as RGB. No contrast enhancement, hand-tuned crops, external
   rescaling or dental fine-tuning. Use each official model's native preprocessing.
3. Encode each full image once, then prompt each tooth independently with the same
   original-coordinate bounding box. No point or mask prompt; no cross-tooth memory.
4. Use FP32, `multimask_output=False`, logit threshold zero, no mask clipping to the
   box, no component selection, no morphology, no SAM 2 dynamic multi-mask fallback.
   No ground truth is used to select predictions or refine masks.
5. Score the returned mask against that tooth's full-resolution ground truth. No
   assignment optimization is needed because prompt/instance correspondence is fixed.
6. **Primary rank:** average Dice over teeth within each image, average the two
   annotation sets of the duplicate image, then average across unique image groups.
   This prevents the duplicated radiograph from receiving twice the weight.
   Also report group-macro IoU, boundary F1 at 2 pixels, tooth-macro and
   area-weighted micro Dice/IoU, precision/recall, per-class scores, empty masks,
   and fraction of teeth with IoU below 0.5. The thresholds are fixed in advance.
7. Bootstrap whole decoded-pixel image groups 5,000 times for 95% confidence intervals. Use the same
   bootstrap draws for paired model differences. Teeth and duplicated annotation
   sets from one radiograph are correlated.
   Paired intervals are pointwise and are not corrected for all 55 comparisons;
   point-estimate ranks do not by themselves establish significant differences.
8. HD95 uses the maximum of the two directed 95th-percentile boundary distances.
   ASSD averages the pooled bidirectional boundary distances. Both are in pixels,
   not millimeters. They are undefined for empty predictions and recorded as null;
   empty counts are always reported, so these cases cannot disappear from Dice/IoU.
   Boundary F1 uses pixel boundary precision/recall, not an area-weighted surface Dice.
9. Time image encoding and per-box decoding separately after a warmup. Inference time
   includes native preprocessing, excludes file reads, model loading, metrics and
   result writes. CUDA timing synchronizes the device. Record peak process memory
   and CUDA allocation. This is a shared server, so timing is indicative, not a
   dedicated-hardware throughput claim. Compare hardware and precision in separate suites.
10. Missing or crashed images make a model ineligible for ranking until the run
    completes. A valid empty mask is retained with Dice and IoU zero. The suite
    reports all expected models and refuses to combine different protocols/cohorts.

The 11 variants are SAM 1 ViT-B/L/H; SAM 2 Hiera tiny/small/base-plus/large; and
SAM 2.1 Hiera tiny/small/base-plus/large. Each checkpoint is its own ranking entry:
comparing only generation names would hide substantial model-size differences.
SAM 2 and 2.1 use the corresponding official configs and distinct weights in the
same pinned SAM 2 code revision. No unofficial dental adaptations are substituted.

## Commands on the server

```bash
cd /data/teeth-sam-benchmark

# Verify the core mask/metric invariants.
.venv/bin/python -m pytest -q test_core.py test_aggregate.py

# Small development check: never use this as the final model ranking.
nice -n 10 .venv/bin/python -u suite.py --models sam2_tiny sam21_tiny sam1_vit_b \
  --split dev --limit 2 --device cpu --threads 2 --output results/smoke_cpu

# Full held-out evaluation, all 11 models; restart the same command to resume.
nohup nice -n 10 .venv/bin/python -u suite.py --models all \
  --split test --condition exact --device cpu --threads 2 \
  --output results/test_exact_cpu > logs/test_exact_cpu.log 2>&1 < /dev/null &

# Show progress and regenerate a partial or complete report.
python3 status.py results/test_exact_cpu
.venv/bin/python aggregate.py --suite results/test_exact_cpu
tail -n 20 logs/test_exact_cpu.log

# Inspect one completed image. Use an image ID from that run's run.json.
.venv/bin/python preview.py --run results/smoke_cpu/sam1_vit_b --image-id IMAGE_ID
```

Results are resumable only when the complete run configuration, code hashes,
dataset manifest, model checkpoint, hardware and precision match. Use a different
output directory after changing any of these. Atomic image writes prevent a partial
file being treated as completed. Process locks prevent duplicate writers.

For a GPU host, copy the project and data, create a new environment with a matching
CUDA PyTorch build, install the pinned repositories, and use `--device cuda` with a
new output directory. Do not mix those timing results with the CPU suite.

## Box robustness experiments

Run after the exact-box baseline, in separate suites. `pad5` and `pad10` expand
each side by 5% or 10% of the box width/height; `jitter5` and `jitter10` independently
perturb each edge by a uniform ±5% or ±10% of the corresponding dimension. Boxes
are clipped to image bounds. Jitter is keyed by seed and instance ID and therefore
identical across models regardless of processing order. Use at least three seeds
for jitter, e.g. 20260909, 20260910 and 20260911; report seeds separately or average
seed replicates within each image before bootstrapping. The included aggregator
keeps seeds separate to prevent accidental pseudoreplication.

```bash
.venv/bin/python suite.py --models all --split test --condition pad5 \
  --device cuda --output results/test_pad5_cuda
.venv/bin/python suite.py --models all --split test --condition jitter5 \
  --seed 20260909 --device cuda --output results/test_jitter5_seed20260909_cuda
```

These perturbations study sensitivity to box quality; they do not establish
performance with actual detector errors, missed teeth or false positive boxes.

## Reproduction and limits

`requirements.lock.txt` captures the installed environment and `source_revisions.json`
captures repository commits. Install the same PyTorch wheel family and source
commits; use `SAM2_BUILD_CUDA=0 pip install --no-build-isolation --no-deps -e repos/sam2`
on CPU after installing its dependencies. No system Python packages were changed.

The archive is CC0. Retain attribution to Humans in the Loop and the original
panoramic radiography source described on Kaggle. SAM checkpoint licenses remain
with their upstream projects. Zero-shot here means no dental fine-tuning by this
benchmark; overlap with foundation-model pretraining cannot be established from
this dataset alone. A single annotation set has labeling error and cannot provide
a universal or clinical ranking. Inspect overlays, audit duplicate/patient identity
where possible, and externally validate before making broader claims.

Sources:

- [Dataset and annotation description](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images/data)
- [Official SAM 1 repository and checkpoints](https://github.com/facebookresearch/segment-anything)
- [Official SAM 2 repository, SAM 2.1 updates and checkpoints](https://github.com/facebookresearch/sam2)
- [COCO mask utilities and polygon/RLE semantics](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)
