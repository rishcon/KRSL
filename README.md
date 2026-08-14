# KRSL AI

Local-first research and MVP project for recognition of isolated
Kazakh-Russian Sign Language (KRSL) signs. The first milestone is a
reproducible signer-independent KRSL20 recognizer, not continuous sign-language
translation.

## Current status

Phase 2 is in progress: versioned video-to-landmark preprocessing. The project
foundation, legacy audit, and signer-independent data inventory are complete;
no model training code is included yet.

## Local setup

Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

Raw videos and extracted keypoints are intentionally excluded from Git. See
`STATUS.md` for the active phase and runnable commands.

## Phase 1 inventory

```powershell
python scripts/build_krsl20_manifest.py `
  --video-root .\V2_videos_5signers_isolated_signs\V2_videos_5signers_isolated_signs `
  --keypoint-root .\V2_keypoints_5signers_isolated_signs\V2_keypoints_5signers_isolated_signs `
  --manifest .\data\manifests\krsl20_v1.csv `
  --report .\reports\dataset_report.json `
  --split-config .\splits\krsl20_signer_independent_v1.json
```

## Phase 2 single-video extraction

Download the MediaPipe Holistic model to `models/holistic_landmarker.task`, then
run:

```powershell
python scripts/extract_holistic_video.py `
  --video .\path\to\sample.mov `
  --model .\models\holistic_landmarker.task `
  --output .\artifacts\smoke\one_video.npz
```

Batch extraction uses the committed manifest and keeps its cache and failure log
local:

```powershell
python scripts/extract_holistic_batch.py `
  --manifest .\data\manifests\krsl20_v1.csv `
  --video-root .\V2_videos_5signers_isolated_signs\V2_videos_5signers_isolated_signs `
  --model .\models\holistic_landmarker.task `
  --output-root .\data\processed\holistic-v1 `
  --failure-log .\reports\processing\holistic-v1_failures.jsonl `
  --summary .\reports\processing\holistic-v1_summary.json
```
