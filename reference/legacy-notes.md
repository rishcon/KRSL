# Legacy reference audit: `kurshakuz/krsl-recogniton`

Audited from the public `master` branch on 2026-08-14. This directory contains
notes only; the legacy repository is not vendored or imported by production
code.

## Reusable knowledge

- The project is Apache License 2.0 licensed. Retain notices and attribution if
  source is copied; its license does not grant rights to third-party datasets.
- `sign-prediction/train_utils.py` uses a fixed sequence shape of **70 × 84**.
  Text landmark vectors are padded with zeroes to 25,200 values.
- Its baseline model is seven sequential `LSTM(64)` layers (the first six return
  sequences), followed by a softmax dense classifier.
- `sign-prediction/train.py` trains for 250 epochs with batch size 16 and saves
  `model.h5`.
- `sign-prediction/predict.py` loads `model.h5` and maps predictions through
  the 20 labels in `sign-prediction/label.txt`.
- Top-level `recognition.py` documents an end-to-end video-to-text path.

## Compatibility and quality findings

- The split is created inside each class by file order (`k % 9 == 1`); it is
  not signer-independent and must not be used to report product quality.
- The legacy Docker image is based on Ubuntu 18.04, TensorFlow 1.14.0,
  OpenJDK 8, and Bazel 3.4.1. Its requirements also constrain OpenCV to
  `<4.0.0`.
- It vendors a large MediaPipe/Bazel tree and invokes shell commands with
  `os.system`; do not reuse this stack in production.
- The prediction threshold suppresses output below 50%, but it does not expose
  a calibrated `UNKNOWN` class or abstention contract.

## Decision

Phase 3 may reproduce the 70 × 84 LSTM as a separately implemented baseline.
All new preprocessing, splitting, evaluation, and inference code remains in
`src/krsl_ai` and uses supported dependencies.

## Sources

- <https://github.com/kurshakuz/krsl-recogniton>
- <https://github.com/kurshakuz/krsl-recogniton/blob/master/sign-prediction/train_utils.py>
- <https://github.com/kurshakuz/krsl-recogniton/blob/master/Dockerfile>
