@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe scripts\train_transformer.py ^
  --manifest data\manifests\krsl20_v1.csv ^
  --sequence-root data\interim\training-sequence-v2 ^
  --report-dir reports\experiments\transformer-v1 ^
  --epochs 30 ^
  --batch-size 32
pause
