#!/usr/bin/env bash
# Install into this checkout; SAM source revisions come from source_revisions.json.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v git >/dev/null
command -v curl >/dev/null
if [[ ! -x .venv/bin/python ]]; then
  "${SAMBENCH_PYTHON:-python3}" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert (3, 12) <= sys.version_info[:2] < (3, 14), "Use Python 3.12 or 3.13 (set SAMBENCH_PYTHON before setup)."'
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cpu

mkdir -p repos
checkout_pinned() {
  local repo_name="$1" repo_url="$2" revision
  revision=$(.venv/bin/python -c 'import json,sys; print(json.load(open("source_revisions.json"))[sys.argv[1]])' "$repo_name")
  if [[ ! -e "repos/$repo_name" ]]; then
    git clone --filter=blob:none "$repo_url" "repos/$repo_name"
    git -C "repos/$repo_name" checkout --detach "$revision"
  fi
  if [[ "$(git -C "repos/$repo_name" rev-parse HEAD)" != "$revision" ]] || \
     [[ -n "$(git -C "repos/$repo_name" status --porcelain --untracked-files=no)" ]]; then
    echo "repos/$repo_name differs from its pinned revision; use a fresh checkout for setup." >&2
    exit 1
  fi
}
checkout_pinned segment-anything https://github.com/facebookresearch/segment-anything.git
checkout_pinned sam2 https://github.com/facebookresearch/sam2.git
.venv/bin/python -m pip install --no-build-isolation --no-deps -e repos/segment-anything
SAM2_BUILD_CUDA=0 .venv/bin/python -m pip install --no-build-isolation --no-deps -e repos/sam2
.venv/bin/python -c 'import torch, torchvision, segment_anything, sam2; print("SAMBench CPU setup ready; PyTorch", torch.__version__)'
echo "Next: download data and checkpoints using the README."
