# Two-mode SAM benchmark CLI

Run commands from your checkout (examples use `/data/teeth-sam-benchmark`).
Use the project's `.venv/bin/python`. For installation and data download, see [README.md](README.md).

## Tiny example: both benchmarks, 10 images

```bash
cd /data/teeth-sam-benchmark

nice -n 10 .venv/bin/python benchmark.py run \
  --mode both \
  --family sam2.1 --size tiny \
  --samples 10 --seed 20260909 \
  --device cpu --threads 2 \
  --output results/tiny_both_10 \
  --background
```

This launches two separate evaluations of SAM 2.1 tiny on the same 10 test images.
Remove `--background` to see progress in the terminal. Change `--samples 10` to
`--samples 2` for two images; use a different output folder when changing arguments.
`--samples 0` (the default) evaluates all 535 eligible test images. The count is
**images, not teeth**. A seed controls deterministic sampling. Duplicate-image
annotation sets are never separated; all models and both modes share one saved cohort.

## Choose a benchmark

| Argument | What the model receives | How predictions are scored |
|---|---|---|
| `--mode auto` | Full image and SAM's automatically generated uniform point grid; no GT prompts | All generated masks are matched one-to-one to GT teeth; extra masks are false positives and unmatched teeth are false negatives |
| `--mode bbox` | Full image plus one GT-derived bounding box per tooth | One returned mask is compared directly with the corresponding GT tooth |
| `--mode both` | Runs both protocols separately | Two separate leaderboards on the same sampled images |

Replace `--mode both` in the example with `auto` or `bbox` to run only that mode.
Auto mode has no tooth classifier and generates general object proposals. Proposals
for bone, background structures, and duplicate teeth all count as false positives.
Ground truth is never used to filter, select top-k, or refine automatic predictions.

## Choose models and sizes

```bash
.venv/bin/python benchmark.py models

# SAM 2.1 tiny versus SAM 2.1 large versus SAM 1 huge
nice -n 10 .venv/bin/python benchmark.py run \
  --mode both \
  --models sam21_tiny sam21_large sam1_vit_h \
  --samples 10 --seed 20260909 \
  --device cpu --threads 2 \
  --output results/tiny_large_huge_10 \
  --background

# All 11 downloaded checkpoints, on a development subset
.venv/bin/python benchmark.py run \
  --mode both --models all --split dev --samples 5 \
  --output results/all_models_dev5 --background
```

| Family | Valid `--size` values | Explicit IDs |
|---|---|---|
| `sam1` | `base`, `large`, `huge` | `sam1_vit_b`, `sam1_vit_l`, `sam1_vit_h` |
| `sam2` | `tiny`, `small`, `base_plus`, `large` | `sam2_tiny`, `sam2_small`, `sam2_base_plus`, `sam2_large` |
| `sam2.1` | `tiny`, `small`, `base_plus`, `large` | `sam21_tiny`, `sam21_small`, `sam21_base_plus`, `sam21_large` |

Use either `--family ... --size ...` for one model or `--models ...` for explicit
selections. “Huge” means SAM 1 ViT-H; SAM 2/2.1 have no huge checkpoint. Mixed
generation/size comparisons are permitted, but every entry retains its full ID.

## Inspect, resume and export

```bash
.venv/bin/python benchmark.py status --output results/tiny_both_10
tail -f results/tiny_both_10/suite.log

# Resume after interruption using the exact saved settings; skip completed images.
.venv/bin/python benchmark.py resume --output results/tiny_both_10 --background

# Rebuild current reports, including coverage of incomplete jobs.
.venv/bin/python benchmark.py report --output results/tiny_both_10

# Validate an experiment and display its sample IDs without running or writing files.
.venv/bin/python benchmark.py run --mode both --family sam2.1 --size tiny \
  --samples 10 --dry-run

.venv/bin/python benchmark.py doctor
.venv/bin/python benchmark.py run --help
```

To stop a background run while preserving completed images:

```bash
kill -TERM "$(.venv/bin/python -c 'import json; print(json.load(open("results/tiny_both_10/launch.json"))["pid"])')"
```

Foreground Ctrl-C stops the active worker and preserves completed images. A process
lock prevents duplicate executions of one output directory. For background runs,
`launch.json` records the controller PID. Results are saved atomically per image.
Resuming refuses changed code/data/configuration or hardware; use a new output
directory after changing the protocol. Missing/corrupt checkpoints and invalid
model/size/sample combinations produce errors. `source_snapshot/` preserves code.

Outputs:

- `plan.json`: exact models, modes, cohort, sample seed, checkpoint/source/data hashes
  and all settings. This defines the experiment.
- `REPORT.md`: separate automatic and box-prompt tables and interpretation.
- `comparison.csv`, `coverage.csv`, `summary.json`, `paired_differences.csv`:
  machine-readable combined views; comparisons retain the mode label.
- `auto/leaderboard.csv` and `bbox/leaderboard.csv`: mode-specific rankings.
- `<mode>/per_image.csv`, `per_tooth.csv`, `per_class.csv`: detailed scores.
- `<mode>/<model>/images/*.json`: every returned mask in COCO RLE, model confidence,
  GT-instance scores and, for auto, matches and false-positive indices.
- `<mode>/<model>/inference.log`, `runtime.json`, `model_info.json`, `status.json`:
  execution evidence, timings, parameter count and peak memory.

## Evaluation definitions

Automatic matching uses mask IoU ≥ 0.5 by default. Hungarian assignment first
maximizes the number of valid one-to-one matches, then their total IoU. A duplicate
mask cannot match the same tooth twice. The primary automatic score is
`sum(matched IoU) / (TP + 0.5 FP + 0.5 FN)`, called **instance quality**. It penalizes
missed teeth and extra masks. This is PQ-style instance scoring, not formal panoptic
quality, because masks may overlap. The report includes detection precision/recall/F1,
FP/FN counts, and GT-macro Dice/IoU/boundary F1 with zero scores for unmatched teeth.

Official COCO mask AP (0.50:0.05:0.95), AP50, AP75 and AR100 are also reported.
Evaluation is class-agnostic with a single “tooth” category; confidence is SAM's
predicted IoU. COCO uses its standard 100-detections-per-image cap. Instance-quality
scoring retains every returned proposal. No mask is selected by its GT overlap
before evaluation. Conventional COCO AP weights image records; primary metrics
instead balance identical-pixel image groups.

Bbox mode retains the previous protocol: original-resolution images, GT-derived
boxes, FP32, one mask per prompt, threshold zero and no optional morphological
cleanup or SAM 2 dynamic multi-mask fallback. It ranks group-macro Dice and reports
IoU and boundary F1. Automatic and bbox scores are not pooled into one ranking.
An automatic failure to find a tooth and a valid empty bbox mask score zero.
Runtime failures make a job incomplete and ineligible for ranking.

All primary scores average within image, then within exact decoded-image duplicate
groups, then across groups. Confidence intervals use 5,000 bootstrap group resamples
by default. Paired differences are within each mode and have pointwise 95% intervals,
without adjustment for multiple comparisons. Very small subsets are implementation
checks, not reliable evidence of model superiority.

## Automatic generator settings

The common defaults are a 32×32 point grid, 8 points per batch, predicted-IoU
threshold 0.88, stability threshold 0.95, NMS threshold 0.7, and no extra crop layers.
The same explicit settings are applied to all selected models; they are not each
family's individually optimized native defaults. Both use native multimask proposal
generation, and SAM 2 mask-to-mask refinement is off.

```bash
# Faster implementation check: 8×8 grid, one development image.
# This changes the automatic protocol and is not comparable to a 32×32 run.
.venv/bin/python benchmark.py run --mode auto --family sam2.1 --size tiny \
  --split dev --samples 1 --points-per-side 8 --points-per-batch 8 \
  --output results/auto_quick_check
```

Available controls: `--points-per-side`, `--points-per-batch`, `--pred-iou-thresh`,
`--stability-thresh`, `--nms-thresh`, `--crop-layers`, `--match-iou`. Tune choices
on development images and freeze them before testing. Increasing grid density or
crop layers increases compute. `--samples` changes the cohort size, while
`--points-per-side` changes automatic inference behavior.

For bbox robustness, use `--box-condition exact|pad5|pad10|jitter5|jitter10`. This
option only affects bbox mode; pure auto rejects a non-default box condition.

Timings are native API wall times after warmup, excluding evaluation and file I/O.
Automatic timings include its grid/crops, filtering/NMS and native RLE output.

## GPU setup

The validated environment is Linux CPU with Python 3.12 and PyTorch 2.7.1.
On a separate CUDA-capable checkout, complete the README setup, then replace the
CPU wheels before starting an experiment. For an NVIDIA driver supporting CUDA 12.8:

```bash
.venv/bin/python -m pip install --upgrade \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python benchmark.py doctor
```

Other wheel variants are listed in the [official PyTorch 2.7.1 instructions](https://pytorch.org/get-started/previous-versions/#v271).
Use `--device cuda` and a new output directory; CPU and GPU timing results belong
in separate experiments. The optional SAM 2 CUDA extension is not required by this
protocol, which disables its component cleanup. GPU execution has not been tested
on the original CPU host.

## Dataset and legacy experiment

See the [dataset audit and original box-only protocol](docs/DATASET_PROTOCOL.md)
for annotation exclusions, duplicate handling, legacy `suite.py` commands, and
dataset limitations. The full original environment freeze is retained in
`requirements.lock.txt`; setup uses the direct dependencies in `requirements.txt`
and the pinned source checkouts, so the editable SAM packages live in `repos/`.

Sources: [SAM automatic mask generator](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py),
[SAM 2 automatic mask generator](https://github.com/facebookresearch/sam2/blob/main/sam2/automatic_mask_generator.py),
[official COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).
