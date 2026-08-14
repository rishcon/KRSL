# KRSL AI status

## Active phase

**Phase 1 — KRSL20 inventory and signer-independent splits**

## Completed

- Connected local `main` to `https://github.com/rishcon/KRSL`.
- Created a modern Python package scaffold and quality tooling.
- Added Git exclusions for raw videos, keypoints, preprocessing caches, and
  experiment/model artifacts.
- Audited the legacy reference and recorded the result in
  `reference/legacy-notes.md`.
- Passed Phase 0 checks on Python 3.12.4: `ruff format --check .`,
  `ruff check .`, and `pytest` (1 passed).

## Commands

```powershell
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

## Next phase gate

- Inventory the local KRSL20 video/keypoint directories without modifying raw
  material.
- Determine signer identifiers and create reproducible signer-independent
  train/validation/test split manifests.

## Known constraints

- The repository contains local raw source material under
  `V2_videos_5signers_isolated_signs/` and
  `V2_keypoints_5signers_isolated_signs/`; it must not be uploaded to Git.
- KRSL20 data-use rights must be confirmed before any commercial release.
