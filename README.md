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
