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

## Embedding reuse and timing

All families cache image features for every box and automatic point batch. In
`--mode both`, one worker loads a model once, then processes each image's bbox
evaluation followed by automatic evaluation using the same full-image embedding.
Only image features are shared; automatic prompts remain independent of GT.
With default `--crop-layers 0`, an image with 30 teeth needs **one encoding across
both modes**, plus the box decodes and point-grid decodes. Each additional crop
needs its own encoding; one extra layer adds four crop encodings. Features are
released between images and after failures, so memory does not grow with the cohort.
Warmup separately encodes one image per worker and is excluded from measurements.

Per-image JSON and CSV record `encoder_calls`, `embedding_reuses`,
`encode_seconds`, and `shared_encode_seconds`. `inference_seconds` includes
the encoding cost in each mode so a mode does not appear faster merely because
the other mode ran first. `actual_inference_seconds` measures the work actually
performed; sum it across modes for inference time without double counting shared
encoding. It excludes warmup, model loading, metrics, input reads and file writes.
Model load time and peak memory are shared worker measurements. Automatic timing
still includes native grid/crop decoding, filtering/NMS and RLE conversion.

Results are saved separately after each image/mode. On resume, completed modes
are skipped, and an image is encoded again only if a missing mode needs it; no
embeddings are persisted to disk. Use a new output directory after this code upgrade.
No additional optimization flag is required.

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

# All 11 ungated checkpoints, on a development subset
.venv/bin/python benchmark.py run \
  --mode both --models ungated --split dev --samples 5 \
  --output results/all_models_dev5 --background
```

| Family | Valid `--size` values | Explicit IDs |
|---|---|---|
| `sam1` | `base`, `large`, `huge` | `sam1_vit_b`, `sam1_vit_l`, `sam1_vit_h` |
| `sam2` | `tiny`, `small`, `base_plus`, `large` | `sam2_tiny`, `sam2_small`, `sam2_base_plus`, `sam2_large` |
| `sam2.1` | `tiny`, `small`, `base_plus`, `large` | `sam21_tiny`, `sam21_small`, `sam21_base_plus`, `sam21_large` |
| `sam3` | `default` (or omit `--size`) | `sam3` |

Use either `--family ... --size ...` for one model or `--models ...` for explicit
selections. “Huge” means SAM 1 ViT-H; SAM 2/2.1 have no huge checkpoint. Mixed
generation/size comparisons are permitted, but every entry retains its full ID.

## SAM 3

The `sam3` entry evaluates the **visual instance-segmentation (PVS/tracker) head**
of Meta's `facebook/sam3` checkpoint using `transformers==5.17.0`. It supports CPU
FP32 and CUDA. This is the [SAM 3 interface for boxes and points](https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/sam3_tracker),
which corresponds to the existing SAM 1/2 task. No text, concept exemplar, or
ground-truth mask is supplied. SAM 3 has one entry; Tiny/Large/Huge are not SAM 3 variants.

```bash
.venv/bin/python -m pip install -r requirements-sam3.txt
.venv/bin/python -c "from huggingface_hub import login; login(add_to_git_credential=False)"
.venv/bin/python download_checkpoints.py --models sam3

nice -n 10 .venv/bin/python benchmark.py run \
  --mode both --models sam3 --samples 50 --device cpu --threads 2 \
  --output results/sam3_both_50 --background

# All 12 entries, including SAM 3; all must have their checkpoints installed
nice -n 10 .venv/bin/python benchmark.py run \
  --mode both --models all --samples 50 --device cpu --threads 2 \
  --output results/all12_both_50 --background
```

Use a read token from an account with approved access to `facebook/sam3`. Enter it
only at the hidden login prompt. The downloader uses Hugging Face's cached login;
tokens are never saved in benchmark plans, checkpoint metadata, or Git.

The Hub revision is pinned in `models.json`. The downloaded weights, model config,
and processor config each have SHA256 provenance; inference loads them locally.
Every required tracker weight must load successfully. Image embeddings are cached
once per image/crop. Box mode returns one mask with dynamic multi-mask fallback,
hole filling, and sprinkle removal disabled, using logit threshold zero.

For automatic mode, the existing pinned **SAM 2 automatic mask generator** supplies
the grid, crop schedule, stability/quality filters, and NMS. Its predictor is replaced
with the SAM 3 adapter: every mask logit and quality score comes from SAM 3. This
keeps proposal processing common across SAM 2/2.1/3, with each model's native image
size and preprocessing. Reports identify the different backend; timings include the
adapter and native Hugging Face calls. This does not evaluate SAM 3 concept/text segmentation.

`--models ungated` selects only SAM 1/2/2.1 and needs no Hugging Face login.
`--models all` selects all 12 entries and requires SAM 3 setup. The downloader defaults
to `ungated`. Optional adapter tests use synthetic outputs and need no gated weights:
`.venv/bin/python -m pytest -q test_sam3.py`.

Upgrading changes source hashes. Keep the previous checkout/environment to resume
old experiments, or restore their exact source snapshots; use a new output directory
for new code. Do not combine an old run and a new SAM 3 run into one leaderboard.

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
