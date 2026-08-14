@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe scripts\train_lstm_baseline.py ^
  --manifest data\manifests\krsl20_v1.csv ^
  --sequence-root data\interim\training-sequence-v1 ^
  --report-dir reports\experiments\lstm-baseline ^
  --epochs 30 ^
  --batch-size 32
pause
