@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "PYTHON=.venv\Scripts\python.exe"
set "VIDEO_ROOT=V2_videos_5signers_isolated_signs\V2_videos_5signers_isolated_signs"
set "KEYPOINT_ROOT=V2_keypoints_5signers_isolated_signs\V2_keypoints_5signers_isolated_signs"
set "MODEL=models\holistic_landmarker.task"

echo [1/5] Checking Python environment...
if not exist "%PYTHON%" (
  py -3.12 -m venv .venv
  if errorlevel 1 goto :error
  "%PYTHON%" -m pip install --upgrade pip
  if errorlevel 1 goto :error
  "%PYTHON%" -m pip install -e ".[dev,vision]"
  if errorlevel 1 goto :error
)

echo [2/5] Checking MediaPipe model...
if not exist "%MODEL%" (
  if not exist models mkdir models
  curl -L --fail --output "%MODEL%" "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
  if errorlevel 1 goto :error
)

echo [3/5] Building KRSL20 manifest and signer-independent split...
"%PYTHON%" scripts\build_krsl20_manifest.py ^
  --video-root "%VIDEO_ROOT%" ^
  --keypoint-root "%KEYPOINT_ROOT%" ^
  --manifest data\manifests\krsl20_v1.csv ^
  --report reports\dataset_report.json ^
  --split-config splits\krsl20_signer_independent_v1.json
if errorlevel 1 goto :error

echo [4/5] Extracting holistic-v1 features. Existing valid artifacts will be skipped.
"%PYTHON%" scripts\extract_holistic_batch.py ^
  --manifest data\manifests\krsl20_v1.csv ^
  --video-root "%VIDEO_ROOT%" ^
  --model "%MODEL%" ^
  --output-root data\processed\holistic-v1 ^
  --failure-log reports\processing\holistic-v1_failures.jsonl ^
  --summary reports\processing\holistic-v1_summary.json
if errorlevel 1 goto :error

echo [5/5] Building normalized 70-step training sequences.
"%PYTHON%" scripts\build_training_sequences.py ^
  --manifest data\manifests\krsl20_v1.csv ^
  --artifact-root data\processed\holistic-v1 ^
  --output-root data\interim\training-sequence-v1 ^
  --report reports\processing\training-sequence-v1_summary.json
if errorlevel 1 goto :error

echo.
echo Done. Summaries are in reports\processing\.
pause
exit /b 0

:error
echo.
echo Pipeline stopped with error %errorlevel%.
pause
exit /b 1
