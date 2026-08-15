# KRSL AI status

## Active phase

**Phase 4 — isolated-video inference**

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
- Added `run_krsl_pipeline.bat`: a local launcher that checks the environment
  and model, rebuilds the manifest, and runs resumable batch extraction with
  per-video progress counters.
- Validated all 5,242 `holistic-v1` artifacts: no invalid files; 77,108 total
  frames; median sequence length 13 frames and maximum 60. Phase 3 will pad
  sequences to 70 frames rather than truncate them.
- Phase 3 preparation started: `training-sequence-v1` centers coordinates on
  the shoulders, scales by shoulder width, selects compact landmarks, creates
  70-step sequences, and excludes manifest rows whose folder conflicts with
  their label.
- Added a CPU/GPU-aware bidirectional LSTM baseline. A one-epoch smoke run
  completed with checkpoint and metrics output; the 30-epoch training launcher
  is `train_lstm_baseline.bat`.
- Added a compact masked Transformer experiment for direct comparison on the
  same `training-sequence-v2` data and signer-independent splits.
- The Transformer reached 38.18% held-out accuracy and 30.84% macro-F1, so the
  velocity BiLSTM remains the primary checkpoint at 41.71% and 38.56%.
- Added single-video inference with the same velocity feature pipeline, the
  published BiLSTM checkpoint, top-3 candidates, and calibrated `UNKNOWN` at
  confidence below 60%. The launcher is `recognize_video.bat`.
- Passed the inference smoke test on `P1_gde (1).mov`: the complete pipeline
  ran successfully and returned `gdeQ` at 79.74% confidence. The source label
  is `gde`, so this sample also documents a genuine model classification error.

## Commands

```powershell
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

## Next phase gate

- Test isolated-video inference on newly recorded clips from a signer who is
  absent from the training set and record both accepted and `UNKNOWN` cases.
- Add a landmark overlay for visual inspection before starting webcam mode.

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
- The current primary model has 41.71% held-out accuracy and can be confidently
  wrong; the 60% threshold reduces forced guesses but is not a safety guarantee.
