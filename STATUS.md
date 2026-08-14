# KRSL AI status

## Active phase

**Phase 2 — video to landmarks preprocessing**

## Completed

- Connected local `main` to `https://github.com/rishcon/KRSL`.
- Created a modern Python package scaffold and quality tooling.
- Added Git exclusions for raw videos, keypoints, preprocessing caches, and
  experiment/model artifacts.
- Audited the legacy reference and recorded the result in
  `reference/legacy-notes.md`.
- Passed Phase 0 checks on Python 3.12.4: `ruff format --check .`,
  `ruff check .`, and `pytest` (1 passed).
- Created `data/manifests/krsl20_v1.csv`, `reports/dataset_report.json`, and
  `splits/krsl20_signer_independent_v1.json` from the immutable local data.
- Verified the Phase 1 split: P3–P5 train, P2 validation, P1 test; every split
  contains all 20 labels and no signer crosses a split boundary.
- Completed the Phase 2 single-video smoke test with MediaPipe 1.0.0 on
  `P1_gde (1).mov`: saved a `holistic-v1` artifact with 4 decoded frames,
  hands, pose, face, timestamps, and presence masks.
- Added resumable batch extraction from the manifest. Its one-sample smoke test
  completed with `extracted=1`, `failed=0`; cache, artifacts, and failure logs
  remain local.

## Commands

```powershell
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

## Next phase gate

- Implement a versioned extractor for one video and save a feature artifact
  with its source sample ID.
- Preserve missing landmark information with explicit masks and produce an
  overlay for visual inspection.

## Known constraints

- The repository contains local raw source material under
  `V2_videos_5signers_isolated_signs/` and
  `V2_keypoints_5signers_isolated_signs/`; it must not be uploaded to Git.
- KRSL20 data-use rights must be confirmed before any commercial release.
- The raw folders contain 80 duplicate logical IDs: 40 `P4_kotoriyQ` samples
  appear under both `_kotoriy_q` and `_kto_q`, and 40 `P5_zachem` samples
  appear under both `_zachem` and `_zachem_q`. The manifest preserves both
  paths and reports them; no raw file has been renamed, moved, or deleted.
- The 80 colliding files have distinct SHA-256 hashes, so they are not
  byte-identical copies. Their intended labels require annotation review before
  they are used for a quality claim or final training set.
