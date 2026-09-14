# SAMBench

Benchmark **SAM 1, SAM 2, SAM 2.1, and SAM 3** on dental panoramic X-rays (OPGs), with 12 model entries and two separate evaluations:

- **`auto`**: generate masks without ground-truth prompts; match predictions to annotated teeth and penalize missed teeth and extra masks.
- **`bbox`**: give each tooth's ground-truth bounding box to SAM; compare its predicted mask with the annotation.
- **`both`**: run both modes on the same images, with separate leaderboards.

## Setup

Linux, Python **3.12 or 3.13**, Git, and curl are required. Allow about **20 GB** for data, weights, and the environment, plus space for results.

```bash
cd /data  # or another writable directory
git clone https://github.com/mokar2001/SAMBench.git teeth-sam-benchmark
cd teeth-sam-benchmark
bash scripts/setup_cpu.sh
```

The script creates `.venv`, installs CPU PyTorch, and installs both official SAM repositories at the commits in `source_revisions.json`. Set `SAMBENCH_PYTHON=python3.12` if needed. For CUDA setup, see [the CLI guide](CLI_GUIDE.md#gpu-setup).

## Download and prepare data

Use version 1 of [Humans in the Loop's teeth segmentation dataset](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images) (CC0).

```bash
mkdir -p downloads
curl --fail --location --retry 4 --continue-at - \
  --output downloads/teeth.zip \
  'https://www.kaggle.com/api/v1/datasets/download/humansintheloop/teeth-segmentation-on-dental-x-ray-images?datasetVersionNumber=1'
.venv/bin/python unpack.py
.venv/bin/python prepare.py
.venv/bin/python finalize_prepared.py
.venv/bin/python download_checkpoints.py --models ungated
```

If Kaggle requires sign-in, download version 1 from the dataset page and save it as `downloads/teeth.zip`. For a Tiny-only run, download `--models sam21_tiny` instead of `ungated`.

Preparation converts the original JSON polygons into instance masks and exports `prepared/boxes.csv` and `prepared/coco.json`. Validation retains **595 annotated images / 15,313 teeth**, with **60 development and 535 test images**. Degenerate polygons and unannotated images are recorded in `prepared/exclusions.json`; exact image duplicates stay in one split. See the [dataset audit](docs/DATASET_PROTOCOL.md).

## Run benchmarks

```bash
# SAM 1/2/2.1: all 11 ungated models, both modes, 50 test images
nice -n 10 .venv/bin/python benchmark.py run \
  --mode both --models ungated --samples 50 \
  --device cpu --threads 2 \
  --output results/all_models_both_50 --background

# SAM 2.1 Tiny, both modes, 10 test images
.venv/bin/python benchmark.py run \
  --mode both --family sam2.1 --size tiny --samples 10 \
  --device cpu --threads 2 --output results/tiny_both_10
```

`--samples` counts **images**, not teeth; `0` uses the full selected split. Use `--split dev` for tuning, `--mode auto` or `bbox` for a single task, and a new output directory when changing settings. CPU jobs run sequentially; all-model automatic runs can take many hours.

Embeddings are reused automatically for every box and automatic point batch. With `--mode both`, each model loads once and completes both evaluations for each image using one shared full-image embedding. Additional crops each need one encoding. Reports include encoder/reuse counts and actual inference time; comparison timings include encoding in each mode. No extra flag is needed.

| Family | Sizes / model IDs |
|---|---|
| SAM 1 | `sam1_vit_b`, `sam1_vit_l`, `sam1_vit_h` (Base, Large, Huge) |
| SAM 2 | `sam2_tiny`, `sam2_small`, `sam2_base_plus`, `sam2_large` |
| SAM 2.1 | `sam21_tiny`, `sam21_small`, `sam21_base_plus`, `sam21_large` |
| SAM 3 | `sam3` (one checkpoint; visual instance segmentation) |

Choose a subset with, for example, `--models sam21_tiny sam21_large sam1_vit_h`.

## Add SAM 3

Obtain access to [facebook/sam3](https://huggingface.co/facebook/sam3), then log in on the machine running the benchmark. Enter your read-access token only at the hidden terminal prompt.

```bash
.venv/bin/python -m pip install -r requirements-sam3.txt
.venv/bin/python -c "from huggingface_hub import login; login(add_to_git_credential=False)"
.venv/bin/python download_checkpoints.py --models sam3
nice -n 10 .venv/bin/python benchmark.py run \
  --mode both --models sam3 --samples 10 --device cpu --threads 2 \
  --output results/sam3_both_10 --background
```

After setup, **`--models all --samples 50` includes all 12 models**. Use a new output directory. SAM 3 uses the pinned Hugging Face `Sam3TrackerModel` visual head, with no text prompts; automatic masks use the same grid/filter/NMS implementation as SAM 2. Checkpoint/config hashes and backend versions are saved. It has no Tiny/Large variants. See [SAM 3 details](CLI_GUIDE.md#sam-3).

## Progress, stop, resume, and results

```bash
.venv/bin/python benchmark.py status --output results/all_models_both_50
tail -f results/all_models_both_50/suite.log

# Stop the background run; completed images remain saved
kill -TERM "$(.venv/bin/python -c 'import json; print(json.load(open("results/all_models_both_50/launch.json"))["pid"])')"

# Continue with the saved configuration, skipping completed images
nice -n 10 .venv/bin/python benchmark.py resume \
  --output results/all_models_both_50 --background

.venv/bin/python benchmark.py report --output results/all_models_both_50
```

Results include `REPORT.md`, separate `auto/leaderboard.csv` and `bbox/leaderboard.csv`, per-image/tooth scores, predicted masks, timings, and a saved experiment plan. Bbox ranking uses group-macro **Dice**; automatic ranking uses **instance quality** with one-to-one IoU matching, plus COCO mask AP. Reports include bootstrap 95% confidence intervals and exclude incomplete jobs from ranking.

For all options and metric definitions, see [CLI_GUIDE.md](CLI_GUIDE.md) or run `.venv/bin/python benchmark.py run --help`. Test with `.venv/bin/python -m pytest -q test_core.py test_aggregate.py test_benchmark_cli.py`; after installing SAM 3 dependencies, also run `test_sam3.py test_embedding_reuse.py`.

Data, checkpoints, environments, and results are downloaded/generated locally and excluded from Git. Model code and weights retain their upstream licenses: [SAM 1](https://github.com/facebookresearch/segment-anything), [SAM 2 / 2.1](https://github.com/facebookresearch/sam2), and [SAM 3](https://huggingface.co/facebook/sam3).
